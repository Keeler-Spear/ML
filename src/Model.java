import java.util.function.Function;


public abstract class Model {
    protected Function<Double, Double> sigmoid = (z) -> 1 / (1 + Math.exp(-z));

    protected double learningRate = 0.001;
    protected Function<Double, Double>[] basisFunctions;
    protected Matrix weights;
    protected boolean trained = false;

    public Model(double learningRate, Function<Double, Double>[] basisFunctions) {
        this.learningRate = learningRate;
        this.basisFunctions = basisFunctions;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        trained = false;
    }

    public void setBasisFunctions(Function<Double, Double>[] basisFunctions) {
        this.basisFunctions = basisFunctions;
        trained = false;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public Function<Double, Double>[] getBasisFunctions() {
        return basisFunctions;
    }

    public Matrix getWeights() {
        return weights;
    }

    public void train(Matrix x, Matrix y) {
        if (x.getRows() != y.getRows()) {
            throw new IllegalArgumentException("The data does not have one sample for each label!");
        }

        trained = true;
        trainProtected(x, y);
    }

    //Predict a single sample
    public double predict(double[] sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(new Matrix(sample)).getValue(1, 1);
    }

    //Predict a single sample
    public double predict(Matrix sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }
        if (sample.getCols() != 1 && sample.getRows() != 1) {
            throw new IllegalArgumentException("More than one sample was provided!");
        }

        return predictProtected(sample).getValue(1, 1);
    }

    //Predict a single sample
    public Matrix predictMultipleSamples(double[][] sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(new Matrix(sample));
    }

    //Predict a single sample
    public Matrix predictMultipleSamples(Matrix sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(sample);
    }

    //ToDo: Have it return a Function

    //Calculates the values for a function at the provided points using the functions and weights provided.
    protected Matrix buildFunction(Matrix x) {
        Matrix temp = LinearAlgebra.vectorFromColumn(x, 1);
        Matrix y = LinearAlgebra.scaleMatrix(LinearAlgebra.applyFunction(temp, basisFunctions[0]), weights.getValue(1, 1)); //y = [1; 1; ...; 1]

        //Removing the bias from basis functions
        Function[] fncs = new Function[basisFunctions.length - 1];
        for (int i = 0; i < fncs.length; i++) {
            fncs[i] = basisFunctions[i + 1];
        }

        for (int i = 0; i < fncs.length; i++) {
            for (int j = 1; j <= x.getCols(); j++) {
                temp = LinearAlgebra.vectorFromColumn(x, j);
                y = LinearAlgebra.addMatrices(y, LinearAlgebra.applyFunction(temp, fncs[i]), weights.getValue((j - 1) * (fncs.length) + i + 2, 1));
            }
        }

        return y;
    }


    protected abstract Matrix predictProtected(Matrix x);

    protected abstract void trainProtected(Matrix x, Matrix y);

    public void printClassificationReport(Matrix x, Matrix y) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        Matrix yApprox = predictProtected(x);
        Metrics.printClassificationReport(y, yApprox);
    }


}
