import core.sync.mutex;

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

import atmosphere;
import distribution;

immutable lambdaArray      = [0.15, 0.5, 1.4];
immutable betaArray        = [-0.5, 0.05, 0.2, 0.7, 2, 5];
immutable chiArray         = [0.7, 1.8];
immutable psiArray         = [0.1, 0.6, 1.1, 4];
immutable sampleSizeArray  = [1000, 10000];
immutable quantileLeftArg  = 0.02;
immutable quantileRightArg = 0.98;
immutable gridSize         = 50;
immutable eps              = 1e-3;
immutable maxIter          = 1000;
immutable minIter          = 100;
// lambda, beta, chi, psi, sampleSize
alias ParamsTuple          = Tuple!(double, double, double, double, int);
immutable CSVHead = `sampleSize,lambda,beta,chi,psi,algorithm,iterations,time ms,log2Likelihood,betaEst`;

void main()
{
	writeln("Total threads: ", totalCPUs);

	auto fout = File("view/nvmm_test.csv", "w");
	fout.writeln(CSVHead);
	auto paramsTupleArray = 
		cartesianProduct(
			lambdaArray    .dup,
			betaArray      .dup,
			chiArray       .dup,
			psiArray       .dup,
			sampleSizeArray.dup,
			)
		.array;

	foreach(i, paramsTuple; paramsTupleArray.parallel(1))
	{
		immutable lambda     = paramsTuple[0];
		immutable beta       = paramsTuple[1]; 
		immutable chi        = paramsTuple[2];
		immutable psi        = paramsTuple[3];
		immutable sampleSize = paramsTuple[4];
		// GIG quantile function
		auto qf              = new GeneralizedInverseGaussianQuantile!double(lambda, chi, psi);
		// GHyp random number generator
		auto rng             = new GeneralizedHyperbolicRNG!double(rndGen, lambda, beta, chi, psi);
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
			.map!(u => immutable NormalVarianceMeanMixture!double.PDF(beta, u))
			.array;
		// GHyp sample
		immutable sample     = rng
			.take(sampleSize)
			.array
			.assumeUnique;
		//auto lineOut     = appender!string;
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
			GradientLikelihoodMaximization!double, 
			CoordinateLikelihoodMaximization!double,
			))
		{
			app.formattedWrite("%s,%s,%s,%s,%s,%s,", sampleSize, lambda, beta, chi, psi, Algo.stringof);

			static if(__traits(isSame, Algo , CoordinateLikelihoodMaximization))
				immutable maxIter = .maxIter / 16;

			StopWatch sw;
			size_t iterCount;
			auto optimizer = new Algo(pdfs.length, sample.length);
			sw.start;
			try
			{
				optimizer.putAndSetWeightsInProportionToLikelihood(pdfs, sample);
				optimizer.optimize( ///optimization
					(log2LikelihoodPrev, log2Likelihood) 
					{
						iterCount++;
						return iterCount >= maxIter || log2Likelihood - log2LikelihoodPrev <= eps;
					});				
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

		///Special α-parametrized EM algorithms
		foreach(Algo; TypeTuple!(
			NormalVarianceMeanMixtureEM!double, 
			NormalVarianceMeanMixtureEMAndGradient!double, 
			NormalVarianceMeanMixtureEMAndCoordinate!double,
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
				optimizer.optimize( ///optimization
					//tolerance
					(alphaPrev, beta, double log2LikelihoodPrev, double log2Likelihood)
					{
						iterCount++;
						return iterCount >= maxIter || iterCount >= minIter && log2Likelihood - log2LikelihoodPrev <= eps;
					});				
			}
			catch (FeaturesException e)
			{
				sw.stop;
				app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, "FeaturesException", double.nan);
				continue;
			}
			sw.stop;
			app.formattedWrite("%s,%s ms,%s,%s\n", iterCount, sw.peek.msecs, optimizer.log2Likelihood, optimizer.beta);
		}
	}
}

class GeneralizedInverseGaussianCDF(T): NumericCDF!T
{
	this(T lambda, T chi, T psi)
	{
		immutable mu = 0;
		auto pdf = new GeneralizedInverseGaussianPDF!T(lambda, chi, psi);
		immutable expectation = E_GIG!T(lambda, chi, psi);				
		super(pdf, [expectation]);
	}
}

class GeneralizedInverseGaussianQuantile(T) : NumericQuantile!T
{
	this(T lambda, T chi, T psi)
	{
		super(new GeneralizedInverseGaussianCDF!T(lambda, chi, psi), -1000, 1000);	
	}
}
