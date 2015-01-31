import std.algorithm;
import std.datetime;
import std.exception;
import std.format;
import std.parallelism;
import std.random;
import std.range;
import std.stdio;
import std.traits;
import std.typecons;
import atmosphere;
import distribution;

alias F = double;

void main()
{
	writeln("Total threads: ", totalCPUs);
	immutable F[]
		lambdaArray = [0.25, 0.5, 1, 2, 4],
		etaArray = [1.0],
		omegaArray = [0.25, 0.5, 1, 2, 4],
		betaArray = [0.0, 0.25, 0.5, 1, 2, 4];
	immutable sampleSizeArray  = [1000, 10000];
	immutable quantileLeftArg  = 0.01;
	immutable quantileRightArg = 0.99;
	immutable gridSize         = 50;
	immutable CSVHead          = `sampleSize,lambda,eta,omega,beta,algorithm,iterations,time ms,log2Likelihood,betaEst`;
	auto paramsTupleArray = cartesianProduct(
		lambdaArray.dup,
		etaArray.dup,
		omegaArray.dup,
		betaArray.dup,
		sampleSizeArray.dup,
		).array;
	version(Travis)
	{
		paramsTupleArray = paramsTupleArray[0 .. min(8, $)];
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
		immutable eta        = paramsTuple[1]; 
		immutable omega      = paramsTuple[2];
		immutable beta       = paramsTuple[3];
		immutable sampleSize = paramsTuple[4];
		// GIG quantile function
		auto qf              = new ProperGeneralizedInverseGaussianQuantile!F(lambda, eta, omega);
		// GHyp random number generator
		auto rng             = new ProperGeneralizedHyperbolicRNG!F(rndGen, lambda, eta, omega, beta);
		// string appender for output
		auto app             = appender!string;
		// left GIG bound
		immutable begin      = qf(quantileLeftArg);
		// right GIG bound
		immutable end        = qf(quantileRightArg);
		// grid's step
		immutable step       = (end-begin)/gridSize;
		// GIG grid
		immutable grid       = iota(begin, end+step/2, step).array;
		// Normal PDFs for common algorithms
		immutable pdfs       = grid.map!(u => immutable NormalVarianceMeanMixture!F.PDF(beta, u)).array;
		// GHyp sample
		immutable sample     = rng.take(sampleSize).array.assumeUnique;
		scope(success) synchronized
		{
			fout.write(app.data);
			fout.flush;
		}
		synchronized 
			writefln(
				"cpu %s start [ %s / %s ]: GIGBounds = [%8g .. %8g] GHypParams = (lambda= %4g eta= %4g omega= %4g beta= %4g)",
				taskPool.workerIndex+1,
				i+1, 
				paramsTupleArray.length,
				begin,
				end,
				lambda,
				eta,
				omega,
				beta,
				);
		///Common algorithms
		foreach(Algo; TypeTuple!(
			EMLikelihoodMaximization!F,
			GradientLikelihoodMaximization!F,
			CoordinateLikelihoodMaximization!F,
			))
		{
			app.formattedWrite("%s,%s,%s,%s,%s,%s,", sampleSize, lambda, eta, omega, beta, Algo.stringof);
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
			app.formattedWrite("%s,%s,%s,%s,%s,%s,", sampleSize, lambda, eta, omega, beta, Algo.stringof);

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

final class ProperGeneralizedInverseGaussianQuantile(T) : NumericQuantile!T {
	this(T lambda, T eta, T omega) {
		auto cdf = new ProperGeneralizedInverseGaussianCDF!T(lambda, eta, omega);
		super(cdf, -1000, 1000);	
	}
}
