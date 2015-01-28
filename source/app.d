import std.algorithm;
import std.conv;
import std.csv;
import std.datetime;
import std.exception;
import std.file;
import std.format;
import std.functional;
import std.math;
import std.parallelism;
import std.path;
import std.random;
import std.range;
import std.stdio;
import std.traits;

import atmosphere;
import distribution;

alias F = double;
static assert(isFloatingPoint!F);

final class ProperGeneralizedInverseGaussianQuantile(T) : NumericQuantile!T
{
	this(T lambda, T eta, T omega)
	{
		auto cdf = new ProperGeneralizedInverseGaussianCDF!T(lambda, eta, omega);
		super(cdf, -1000, 1000);	
	}
}

immutable lambdaArray      = [0.15, 0.5, 1.4];
immutable betaArray        = [-0.5, 0.05, 0.2, 0.7, 2, 5];
immutable chiArray         = [0.7, 1.8];
immutable psiArray         = [0.1, 0.6, 1.1, 4];
immutable sampleSizeArray  = [1000, 10000];
immutable quantileLeftArg  = 0.02;
immutable quantileRightArg = 0.98;
immutable gridSize         = 50;
immutable eps              = 1e-5;
immutable maxIter          = 10000;
immutable minIter          = 100;
immutable CSVHead          = `sampleSize,lambda,beta,chi,psi,algorithm,iterations,time ms,log2Likelihood,betaEst`;

// lambda, beta, chi, psi, sampleSize
alias ParamsTuple          = Tuple!(F, F, F, F, int);

void main()
{
	writeln("Total threads: ", totalCPUs);

	auto paramsTupleArray = 
		cartesianProduct(
			lambdaArray    .dup,
			betaArray      .dup,
			chiArray       .dup,
			psiArray       .dup,
			sampleSizeArray.dup,
			)
		.array;



	version(Travis)
	{
		paramsTupleArray = paramsTupleArray[0 .. min(16, $)];
		auto fout = stdout;
	}
	else
	{
		auto fout = File("view/nvmm_test.csv", "w");
		fout.writeln(CSVHead);
	}

	foreach(i, paramsTuple; paramsTupleArray.parallel(1))
	{
		immutable lambda     = paramsTuple[0];
		immutable beta       = paramsTuple[1]; 
		immutable chi        = paramsTuple[2];
		immutable psi        = paramsTuple[3];
		immutable sampleSize = paramsTuple[4];
		immutable params = GIGChiPsi!F(chi, psi);
		// GIG quantile function
		auto qf              = new ProperGeneralizedInverseGaussianQuantile!F(lambda, params.eta, params.omega);
		// GHyp random number generator
		auto rng             = new GeneralizedHyperbolicRNG!F(rndGen, lambda, beta, chi, psi);
		// string appender for output
		auto app             = appender!string;
		scope(success) synchronized
		{
			fout.write(app.data);
			fout.flush;
		}
		// left GIG bound
		immutable begin      = qf(quantileLeftArg);
		// right GIG bound
		immutable end        = qf(quantileRightArg);
		// grid's step
		immutable step       = (end-begin)/gridSize;
		// GIG grid
		immutable grid       = iota(begin, end+step/2, step)
			.array;
		// Normal PDFs for common algorithms
		immutable pdfs       = grid
			.map!(u => immutable NormalVarianceMeanMixture!F.PDF(beta, u))
			.array;
		// GHyp sample
		immutable sample     = rng
			.take(sampleSize)
			.array
			.assumeUnique;

		synchronized 
			writefln(
				"cpu %s start [ %s / %s ]: GIGBounds = [%8g .. %8g] GHypParams = (lambda= %4g beta= %4g chi= %4g psi= %4g)",
				taskPool.workerIndex+1,
				i+1, 
				paramsTupleArray.length,
				begin,
				end,
				lambda,
				beta,
				chi,
				psi,
				);

		///Common algorithms
		foreach(Algo; TypeTuple!(
			GradientLikelihoodMaximization!F, 
			CoordinateLikelihoodMaximization!F,
			))
		{
			app.formattedWrite("%s,%s,%s,%s,%s,%s,", sampleSize, lambda, beta, chi, psi, Algo.stringof);
			StopWatch sw;
			size_t iterCount;
			auto optimizer = new Algo(pdfs.length, sample.length);
			sw.start;
			try
			{
				optimizer.put(pdfs, sample);
				while(sw.peek.msecs < 1000)
				{
					iterCount++;
					// See also `optimize` method to handle optimization with tolerance.
					optimizer.eval();				
				}
			}
			catch (FeaturesException e)
			{
				sw.stop;
				app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, "FeaturesException", "-");
				continue;
			}
			sw.stop;
			app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, optimizer.log2Likelihood, "-");
		}

		///Special beta-parametrized EM algorithms
		foreach(Algo; TypeTuple!(
			NormalVarianceMeanMixtureEM!F, 
			NormalVarianceMeanMixtureEMAndGradient!F, 
			NormalVarianceMeanMixtureEMAndCoordinate!F,
			))
		{
			app.formattedWrite("%s,%s,%s,%s,%s,%s,", sampleSize, lambda, beta, chi, psi, Algo.stringof);

			StopWatch sw;
			size_t iterCount;
			auto optimizer = new Algo(grid, sample.length);
			sw.start;
			try 
			{
				optimizer.sample = sample;
				while(sw.peek.msecs < 1000)
				{
					iterCount++;
					// See also `optimize` method to handle optimization with tolerance.
					optimizer.eval();				
				}		
			}
			catch (FeaturesException e)
			{
				sw.stop;
				app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, "FeaturesException", F.nan);
				continue;
			}
			sw.stop;
			app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, optimizer.log2Likelihood, optimizer.beta);
		}
	}
}
