import core.sync.mutex;
import std.algorithm, std.conv, std.csv, std.datetime, std.file, std.format, std.functional, std.math, 
	std.parallelism, std.path, std.range, std.stdio, std.random;

import atmosphere;
import distribution;

void main()
{
	main2();
}

immutable CSVInputHead = "count, nu, alpha, beta, mu, lambda, q95, q99, fileName";
immutable CSVHead = 
	CSVInputHead ~ ", " ~
	"GLM_iter, " "GLM_time, GLM_lh, " ~
	"CLM_iter, " "CLM_time, CLM_lh, " ~
	"NVMME_iter, " "NVMME_time, NVMME_lh, NVMME_alpha, " ~ 
	"NVMMG_iter, " "NVMMG_time, NVMMG_lh, NVMMG_alpha, " ~
	"NVMMC_iter, " "NVMMC_time, NVMMC_lh, NVMMC_alpha, " ~ 
	"unused";
immutable folder = "data/GH";


class GHypCDF: NumericCDF!real
{
	this(real lambda, GHypChiPsi!real params)
	{
		immutable mu = 0;
		with(params)
		{
			auto pdf = new GeneralizedHyperbolicPDF!real(lambda, alpha, beta, delta, mu);
			immutable expectation = E_GHyp!real(lambda, beta, chi, psi);				
			super(pdf, [expectation]);
		}
	}
}

class GHypQuantile : NumericQuantile!real
{
	this(real lambda, GHypChiPsi!real params)
	{
		super(new GHypCDF(lambda, params), -1000, 1000);	
	}
}

void main2()
{
	auto inputs = folder.buildPath("input.csv").readText.csvReader!(string[string])(null).array;

	immutable lambdas = [0.15, 0.5, 1.4];
	immutable betas = [-0.5, 0.05, 0.2, 0.7, 2, 5];
	immutable chis = [0.7, 1.8];
	immutable psis = [0.1, 0.6, 1.1, 4];

	int fc;

	foreach(input; inputs.filter!`a["size"] = "10000"`)
	{
		immutable mu = input["p_beta"].to!double;
		assert(mu == 0);
		immutable lambda = input["p_nu"].to!double;
		immutable params = GHypChiPsi!real(input["p_alpha"].to!double, input["p_mu"].to!double, input["p_lambda"].to!double);
		immutable oq95 = input["q_95"].to!double;
		immutable oq99 = input["q_99"].to!double;
		auto fn = input["p_tex"];
		immutable sample = folder.buildPath("data", fn).readText.splitter.map!(to!double).array;
		assert(sample.length);
		auto minv = sample.reduce!min;
		auto maxv = sample.reduce!max;
		//writeln(minv, maxv);
		auto fout = File(folder.buildPath("data", fn)~".txt", "w");
		auto frng = File(folder.buildPath("data", fn)~".rn", "w");
		scope(exit) fout.close;
		auto pdf = new GeneralizedHyperbolicPDF!real(lambda, params.alpha, params.beta, params.delta, 0);
		auto rng = new GeneralizedHyperbolicRNG!real(rndGen, lambda, params.beta, params.chi, params.psi);
		foreach(i; 0..100000)
		{
			frng.writeln(rng.front);
		}
		foreach(i; 0..1000)
		{
			auto x = minv + i*(maxv-minv)/1000;
			fout.write(x);
			fout.write(" ");
			fout.writeln(pdf(x));
		}
		//writeln(i);
		//writeln(params);
		try
		{
			auto cdf = new GHypCDF(lambda, params);
			//writeln("created");
			//writeln(cdf(10));		
			auto qf = new GHypQuantile(lambda, params);
			//auto l = qf(0.5);
			auto q95 = qf(0.95);
			auto q99 = qf(0.99);
			assert(approxEqual(q95, oq95));
			assert(approxEqual(q99, oq99));
			writeln(lambda, params);
			writefln("q95 = %s oq95 = %s", q95, oq95);
			writefln("q99 = %s oq99 = %s", q99, oq99);
		}
		catch(Exception e)
		{
			writeln(lambda, params);
			writeln(e.msg);
		immutable expectation = E_GHyp!real(lambda, params.beta, params.chi, params.psi);
		immutable variance = V_GHyp!real(lambda, params.beta, params.chi, params.psi);
		auto divs = [-3, -1, 0, 1, 3].map!(x => x * variance + expectation).array;
		divs.writeln;
			fc++;
		}
	}
	writeln("fails count = ", fc);
}

void Main()
{
	writeln("Total threads: ", totalCPUs);
	auto fout = File(folder.buildPath("output.csv"), "w");
	auto ferr = File(folder.buildPath("fails.csv"), "w");
	auto mout = new Mutex;
	auto merr = new Mutex;
	auto mstd = new Mutex;

	StopWatch gsw;
	gsw.start;
	//file
	fout.writeln(CSVHead);
	//failure
	ferr.writeln(CSVInputHead);
	auto inputs = folder.buildPath("input.csv").readText.csvReader!GHInput(null).array;
	//foreach(i, input; inputs)
	foreach(i, input; inputs.parallel(1))
	{
		StopWatch lsw;
		lsw.start;
		auto lineOut = appender!string;
		lineOut.formattedWrite("%s, ", input);
		scope(exit) 
		{
			lsw.stop;
			synchronized(mstd) writefln("cpu %s [ %s / %s ] %s", taskPool.workerIndex+1, i+1, inputs.length, cast(Duration)lsw.peek);
		}
		scope(success) synchronized(mout)
		{
			fout.writeln(lineOut.data);
			fout.flush;
		}
		scope(failure) synchronized(merr)
		{
			ferr.writeln(input);
			ferr.flush;
		}
		immutable begin = 0.1;
		immutable end = input.q99;
		immutable count = 50;
		immutable step = (end-begin)/count;
		immutable eps = 1e-3;
		immutable grid = iota(begin, end+step/2, step).array;
		immutable sample = folder.buildPath("data", input.fileName).readText.splitter.map!(to!double).array;
		immutable pdfs = grid.map!(u => immutable NormalVarianceMeanMixture!double.PDF(input.alpha, u)).array;
		immutable maxIter = 1000;
		immutable minIter = 100;

		///Common algorithms
		foreach(LM; TypeTuple!(
			  GradientLikelihoodMaximization, 
			CoordinateLikelihoodMaximization,
			))
		{
			StopWatch sw;
			size_t iterCount;
			auto optimizer = new LM!double(pdfs.length, sample.length);
			sw.start;
			try
			{
				optimizer.putAndSetWeightsInProportionToLikelihood(pdfs, sample);
				optimizer.optimize( ///optimization
					(log2LikelihoodPrev, log2Likelihood) 
					{
						iterCount++;
						static if(__traits(isSame, LM , CoordinateLikelihoodMaximization))
							auto maxIter = maxIter / 10;
						return iterCount >= maxIter || log2Likelihood - log2LikelihoodPrev <= eps;

					});				
			}
			catch (FeaturesException e)
			{
				sw.stop;
				lineOut.formattedWrite("%s, %s, FeaturesException, ", iterCount, sw.peek.msecs);
				continue;
			}
			sw.stop;
			lineOut.formattedWrite("%s, %s, %s, ", iterCount, sw.peek.msecs, optimizer.log2Likelihood);
		}

		///Special α-parametrized EM algorithms
		foreach(NVMM; TypeTuple!(
			NormalVarianceMeanMixtureEM, 
			NormalVarianceMeanMixtureEMAndGradient, 
			NormalVarianceMeanMixtureEMAndCoordinate))
		{
			StopWatch sw;
			size_t iterCount;
			auto optimizer = new NVMM!double(grid, sample.length);
			sw.start;
			try 
			{
				optimizer.sample = sample;
				optimizer.optimize( ///optimization
					//tolerance
					(alphaPrev, alpha, double log2LikelihoodPrev, double log2Likelihood)
					{
						iterCount++;
						return iterCount >= maxIter || iterCount >= minIter && log2Likelihood - log2LikelihoodPrev <= eps;
					});				
			}
			catch (FeaturesException e)
			{
				sw.stop;
				lineOut.formattedWrite("%s, %s, FeaturesException, -, ", iterCount, sw.peek.msecs);
				continue;
			}
			sw.stop;
			lineOut.formattedWrite("%s, %s, %s, %s, ", iterCount, sw.peek.msecs, optimizer.log2Likelihood, optimizer.alpha);
		}
	}
	gsw.stop;
	writefln("time: %s", cast(Duration)gsw.peek);
}

struct GHInput
{
	size_t count;
	double nu;
	double alpha;
	double beta;
	double mu;
	double lambda;
	double q95;
	double q99;
	string fileName;

	void toString(scope void delegate(const(char)[]) sink, FormatSpec!char fmt) inout
	{
		sink.formatValue(count, fmt);
		sink(", ");
		sink.formatValue(nu, fmt);
		sink(", ");
		sink.formatValue(alpha, fmt);
		sink(", ");
		sink.formatValue(beta, fmt);
		sink(", ");
		sink.formatValue(mu, fmt);
		sink(", ");
		sink.formatValue(lambda, fmt);
		sink(", ");
		sink.formatValue(q95, fmt);
		sink(", ");
		sink.formatValue(q99, fmt);
		sink(", ");
		sink.formatValue(fileName, fmt);
	}
}
