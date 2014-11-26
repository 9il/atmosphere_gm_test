import core.sync.mutex;
import std.algorithm, std.conv, std.csv, std.datetime, std.file, std.format, std.functional, std.math, 
	std.parallelism, std.path, std.range, std.stdio;
import atmosphere;

immutable CSVInputHead = "count, nu, alpha, beta, mu, lambda, q95, q99, fileName";
immutable CSVHead = 
	CSVInputHead ~ ", " ~
	"GLM_iter, " "GLM_time, GLM_lh, " ~
	"CLM_iter, " "CLM_time, CLM_lh, " ~
	"NVMME_iter, " "NVMME_time, NVMME_lh, NVMME_alpha, " ~ 
	"NVMMC_iter, " "NVMMC_time, NVMMC_lh, NVMMC_alpha, " ~ 
	"NVMMG_iter, " "NVMMG_time, NVMMG_lh, NVMMG_alpha, " ~
	"unused";
immutable folder = "data/GH";

void main()
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
			CoordinateLikelihoodMaximization))
		{
			StopWatch sw;
			size_t iterCount;
			auto optimizer = new LM!double(pdfs.length, sample.length);
			sw.start;
			try
			{
				optimizer.put(pdfs, sample);
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
