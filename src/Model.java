import java.util.function.Function;

/**
 * An abstract class representing a machine learning model.
 *
 * @author Keeler Spear
 * @version %I%, %G%
 * @since 1.0
 */
public abstract class Model {
    protected Function<Double, Double> sigmoid = (z) -> 1 / (1 + Math.exp(-z));

    protected double learningRate = 0.001;
    protected Function<Double, Double>[] basisFunctions;
    protected Matrix weights;
    protected boolean trained = false;

    /**
     * Creates a machine learning model object from the data provided.
     *
     * @param learningRate The learning rate the model will use when training.
     * @param basisFunctions The set of basis functions used to fit the model.
     */
    public Model(double learningRate, Function<Double, Double>[] basisFunctions) {
        this.learningRate = learningRate;
        this.basisFunctions = basisFunctions;
    }

    /**
     * Returns learning rate of the model.
     *
     * @return The learning rate of the model.
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets the model's learning rate.
     *
     * @param learningRate The value that the learning rate will be set to.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        trained = false;
    }

    /**
     * Returns the set of basis functions used by the model.
     *
     * @return The set of basis functions used by the model.
     */
    public Function<Double, Double>[] getBasisFunctions() {
        return basisFunctions;
    }

    /**
     * Sets the model's basis functions.
     *
     * @param basisFunctions The set of functions that the model's basis functions will be set to.
     */
    public void setBasisFunctions(Function<Double, Double>[] basisFunctions) {
        this.basisFunctions = basisFunctions;
        trained = false;
    }

    /**
     * Returns the model's weights.
     *
     * @return The model's weights.
     */
    public Matrix getWeights() {
        return weights;
    }

    /**
     * Trains the model using the model's parameters and the data provided.
     *
     * @param X A matrix of data parameters.
     * @param y A vector of data labels.
     * @throws IllegalArgumentException If each sample does not have a label.
     */
    public void train(Matrix X, Matrix y) {
        if (X.getRows() != y.getRows()) {
            throw new IllegalArgumentException("The data does not have one sample for each label! There are " + X.getRows() + " samples and " + y.getRows() + " samples!");
        }

        trained = true;
        trainProtected(X, y, generateW0(X.getCols() * (basisFunctions.length - 1) + 1));
    }

    /**
     * Predicts the label of a single sample.
     *
     * @param sample The parameter values of a single sample
     * @throws IllegalArgumentException If the model is untrained.
     */
    public double predict(double[] sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(new Matrix(sample)).getValue(1, 1);
    }

    /**
     * Predicts the label of a single sample.
     *
     * @param sample The parameter values of a single sample
     * @throws IllegalArgumentException If the model is untrained.
     * @throws IllegalArgumentException If more than one sample is provided.
     */
    public double predict(Matrix sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }
        if (sample.getCols() != 1 && sample.getRows() != 1) {
            throw new IllegalArgumentException("More than one sample was provided!");
        }

        return predictProtected(sample).getValue(1, 1);
    }

    /**
     * Predicts the labels for a set of samples.
     *
     * @param sample The parameter values for a set of samples.
     * @throws IllegalArgumentException If the model is untrained.
     */
    public Matrix predictMultipleSamples(double[][] sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(new Matrix(sample));
    }

    /**
     * Predicts the labels for a set of samples.
     *
     * @param sample The parameter values for a set of samples.
     * @throws IllegalArgumentException If the model is untrained.
     */
    public Matrix predictMultipleSamples(Matrix sample) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        return predictProtected(sample);
    }

    /**
     * Prints the classification report for the model's performance.
     *
     * @param X The set of sample parameters.
     * @param y The set of sample labels.
     * @throws IllegalArgumentException If the model is untrained.
     */
    public void printClassificationReport(Matrix X, Matrix y) {
        if (!trained) {
            throw new IllegalStateException("The model is not trained!");
        }

        Matrix yApprox = predictProtected(X);
        Metrics.printClassificationReport(y, yApprox);
    }

    //ToDo: Have it return a Function

    //Calculates the values for a function at the provided points using the functions and weights provided.
    protected Matrix buildFunction(Matrix X) {
        Matrix temp = LinearAlgebra.vectorFromColumn(X, 1);
        Matrix y = LinearAlgebra.scaleMatrix(LinearAlgebra.applyFunction(temp, basisFunctions[0]), weights.getValue(1, 1)); //y = [1; 1; ...; 1]

        //Removing the bias from basis functions
        Function[] fncs = new Function[basisFunctions.length - 1];
        for (int i = 0; i < fncs.length; i++) {
            fncs[i] = basisFunctions[i + 1];
        }

        for (int i = 0; i < fncs.length; i++) {
            for (int j = 1; j <= X.getCols(); j++) {
                temp = LinearAlgebra.vectorFromColumn(X, j);
                y = LinearAlgebra.addMatrices(y, LinearAlgebra.applyFunction(temp, fncs[i]), weights.getValue((j - 1) * (fncs.length) + i + 2, 1));
            }
        }

        return y;
    }

    //Actual method for predicting a sample's label
    protected abstract Matrix predictProtected(Matrix x);

    //Actual method for tor training the model
    protected abstract void trainProtected(Matrix x, Matrix y, Matrix w0);

    //Generates an initial guess for the model's weights
    protected abstract Matrix generateW0(int n);

    @Override
    public String toString() {
        return weights.toString();
    }
}
