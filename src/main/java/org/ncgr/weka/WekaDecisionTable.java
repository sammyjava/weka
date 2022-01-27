package org.ncgr.weka;

import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
 
/**
 * Run Weka DecisionTable on ARFF files.
 *
 * -S <search method specification>
 *  Full class name of search method, followed
 *  by its options.
 *  eg: "weka.attributeSelection.BestFirst -D 1"
 *  (default weka.attributeSelection.BestFirst)
 *
 * -X <number of folds>
 *  Use cross validation to evaluate features.
 *  Use number of folds = 1 for leave one out CV.
 *  (Default = leave one out CV)
 *
 * -E <acc | rmse | mae | auc>
 *  Performance evaluation measure to use for selecting attributes.
 *  (Default = accuracy for discrete class and rmse for numeric class)
 *
 * -I
 *  Use nearest neighbour instead of global table majority.
 *
 * -R
 *  Display decision table rules.
 *
 * Options specific to search method weka.attributeSelection.BestFirst:
 *
 * -P <start set>
 *  Specify a starting set of attributes.
 *  Eg. 1,3,5-7.
 *
 * -D <0 = backward | 1 = forward | 2 = bi-directional>
 *  Direction of search. (default = 1).
 *
 * -N <num>
 *  Number of non-improving nodes to
 *  consider before terminating search.
 *
 * -S <num>
 *  Size of lookup cache for evaluated subsets.
 *  Expressed as a multiple of the number of
 *  attributes in the data set. (default = 1)
 */
public class WekaDecisionTable {
    /**
     * Main class runs static methods.
     */
    public static void main(String[] args) throws Exception {
	Options options = new Options();
        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

	// REQUIRED
        Option arffFileOption = new Option("a", "arfffile", true, "ARFF input file (required)");
        arffFileOption.setRequired(true);
        options.addOption(arffFileOption);
	// OPTIONAL
	Option testFileOption = new Option("tf", "testfile", true, "test data file in ARFF format");
	testFileOption.setRequired(false);
	options.addOption(testFileOption);
	// ACTIONS
	Option crossValidateOption = new Option("cv", "crossvalidate", false, "run cross-validation");
	crossValidateOption.setRequired(false);
	options.addOption(crossValidateOption);
	Option testOption = new Option("test", "test", false, "run training/testing on datasets");
	testOption.setRequired(false);
	options.addOption(testOption);
	Option optimizeOption = new Option("op", "optimize", false, "optimize model by running noptimize training/testing runs and choosing best");
	optimizeOption.setRequired(false);
	options.addOption(optimizeOption);
	// PARAMETERS
	Option kfoldOption = new Option("k", "kfold", true, "cross-validation k-fold [10]");
	kfoldOption.setRequired(false);
	options.addOption(kfoldOption);
	Option numOptimizeOption = new Option("no", "numoptimize", true, "the number of training/testing runs to optimize the model [10]");
	numOptimizeOption.setRequired(false);
	options.addOption(numOptimizeOption);
	Option debugOption = new Option("D", "debug", false, "toggle on debug mode [false]");
	debugOption.setRequired(false);
	options.addOption(debugOption);
	
	try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("WekaDecisionTable", options);
            System.exit(1);
            return;
        }

	// actions
	final boolean crossValidate = cmd.hasOption("crossvalidate");
	final boolean test = cmd.hasOption("test");
	final boolean optimize = cmd.hasOption("optimize");

	// parameters
	int kfold = 10;
	int numOptimize = 10;
	if (cmd.hasOption("kfold")) kfold = Integer.parseInt(cmd.getOptionValue("kfold"));
	if (cmd.hasOption("numoptimize")) numOptimize = Integer.parseInt(cmd.getOptionValue("numoptimize"));
	boolean debug = cmd.hasOption("debug");

	// Read instances from the input ARFF files
 	Instances data = Util.rearrange(new DataSource(cmd.getOptionValue("arfffile")).getDataSet());

	// do work
	if (crossValidate) {
	    crossValidate(data, kfold, debug);
	} else if (test) {
	    Instances testData = Util.rearrange(new DataSource(cmd.getOptionValue("testfile")).getDataSet());
	    test(data, testData, debug);
	} else if (optimize) {
	    Instances testData = Util.rearrange(new DataSource(cmd.getOptionValue("testfile")).getDataSet());
	    optimize(data, testData, numOptimize, debug);
	}
    }

    /**
     * Run a cross-validation on the data.
     */
    public static void crossValidate(Instances data, int kfold, boolean debug) throws Exception {
	// run the cross-validation
	DecisionTable model = getDecisionTable();
	model.setDebug(debug);
	Evaluation evaluation = new Evaluation(data);
	System.err.println("# cross-validating DecisionTable with kfold="+kfold);
	Random random = new Random(ThreadLocalRandom.current().nextInt());
	evaluation.crossValidateModel(model, data, kfold, random);
	// output
	Util.printResults(evaluation);
    }

    /**
     * Train/Test a model on the given data.
     */
    public static void test(Instances trainingData, Instances testingData, boolean debug) throws Exception {
	DecisionTable model = getDecisionTable();
	model.setDebug(debug);
	Evaluation trainingEvaluation = new Evaluation(trainingData);
	Evaluation testingEvaluation = new Evaluation(testingData);
        System.err.println("# testing/training DecisionTable");
        model.buildClassifier(trainingData);	                   // train
        trainingEvaluation.evaluateModel(model, trainingData);	   // validate
        testingEvaluation.evaluateModel(model, testingData);	   // test
	// output
	System.err.println("# training result on "+trainingData.size()+" instances");
	Util.printResults(trainingEvaluation);
	System.err.println("# testing result on "+testingData.size()+" instances");
	Util.printResults(testingEvaluation);
    }

    /**
     * Optimize the model by training/testing numOptimize times and saving the best one (highest MCC).
     *
     * serialize model
     * weka.core.SerializationHelper.write("/some/where/j48.model", cls);
     *
     * deserialize model
     * Classifier cls = (Classifier) weka.core.SerializationHelper.read("/some/where/j48.model");
     */
    public static void optimize(Instances trainingData, Instances testingData, int numOptimize, boolean debug) throws Exception {
	TreeMap<Double,Classifier> models = new TreeMap<>();
	TreeMap<Double,Evaluation> evaluations = new TreeMap<>();
        System.out.println("# optimizing DecisionTable with numOptimize="+numOptimize);
	for (int round=1; round<=numOptimize; round++) {
	    DecisionTable model = getDecisionTable();
	    model.setDebug(debug);
	    Evaluation testingEvaluation = new Evaluation(testingData);
            model.buildClassifier(trainingData);                     // train
            testingEvaluation.evaluateModel(model, testingData);     // test
            Util.printResults(testingEvaluation);                 // output
            // store
            evaluations.put(testingEvaluation.matthewsCorrelationCoefficient(1), testingEvaluation);
            models.put(testingEvaluation.matthewsCorrelationCoefficient(1), model);
	}
	// output
	Util.printStats(evaluations);
	// save
	System.out.println("# saving best model to file wekads.model");
	SerializationHelper.write("wekads.model", models.lastEntry().getValue());
    }

    /**
     * Create a DecisionTable classifier with standard parameters.
     */
    public static DecisionTable getDecisionTable() {
	DecisionTable model = new DecisionTable();
	model.setNumDecimalPlaces(4);
	return model;
    }
}
