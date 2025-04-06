import java.util.function.Function;

public class NeuralNetwork extends Model{

    //Input * weights -> function -> output
    private class Node extends Model{
        private final double MAX_RAND = 10;
        private final double MIN_RAND = -10;
        private Function<Double, Double> activationFunction;

        public Node(double learningRate, Function<Double, Double> activationFunction, Function<Double, Double>[] basisFunctions) {
            super(learningRate, basisFunctions);
            this.activationFunction = activationFunction;
        }

        @Override
        protected Matrix predictProtected(Matrix x) {
            Matrix y = buildFunction(x);

            //Only works for binary classification
            return LinearAlgebra.applyFunction(y, activationFunction);
        }

        @Override
        protected void trainProtected(Matrix X, Matrix y, Matrix w0) {
            weights = Regression.logisticReg(X, y, w0, learningRate, basisFunctions, false);
        }

        @Override
        protected Matrix generateW0(int n) {
            return LinearAlgebra.randMatrix(n, 1, MIN_RAND, MAX_RAND);
        }
    }

    private class Layer {
        private Node[] layer;

        public Layer(int width) {
            layer = new Node[width];
        }

        public Node[] getLayer() {
            return layer;
        }

        public void setNode(int i, double learningRate, Function<Double, Double> activationFunction, Function<Double, Double>[] basisFunctions) {
            layer[i] = new Node(learningRate, activationFunction, basisFunctions);
        }

        public Matrix fire(Matrix inputs) {
            Matrix outputs = new Matrix (layer.length, 1);

            for (int i = 1; i <= layer.length; i++) {
                outputs.setValue(i, 1, layer[i].predict(inputs));
            }

            return outputs;
        }
    }









    private static final double MAX_RAND = 10;
    private static final double MIN_RAND = 10;

    private Function<Double, Double> activationFunction;
    private NFunction<Double> objectiveFunction;
    private Layer[] neuralNetwork;

    //ToDo: Create a "default" node in this method
    /**
     * Creates a neural network object from the data provided.
     *
     * @param depth The number of hidden layers the neural network will have.
     * @param width The number of nodes each layer will have.
     * @param learningRate The learning rate the model will use when training.
     * @param basisFunctions The set of basis functions used to fit the model.
     * @throws IllegalArgumentException If the learning rate is less than or equal to zero.
     * @throws IllegalArgumentException If each layer does not hae a width.
     * @throws IllegalArgumentException If any layer's width is less than or equal to zero.
     */
    public NeuralNetwork(int depth, int[] width, Function<Double, Double> activationFunction, NFunction<Double> objectiveFunction, double learningRate, Function<Double, Double>[] basisFunctions) {
        super(learningRate, basisFunctions);

        if (depth <= 0) {
            throw new IllegalArgumentException("Depth must be greater than 0!");
        }

        if (width.length != (depth + 2)) {
            throw new IllegalArgumentException("Each layer must have a width!");
        }
        
        for (int i = 0; i < width.length; i++) {
            if (width[i] <= 0) {
                throw new IllegalArgumentException("Every layer's width must be greater than 0!");
            }
        }

        this.activationFunction = activationFunction;
        this.objectiveFunction = objectiveFunction;
        this.learningRate = learningRate;

        buildNeuralNetwork(depth, width);
    }

    private void buildNeuralNetwork(int depth, int[] width) {
        neuralNetwork = new Layer[depth + 2];

        for (int i = 0; i < depth; i++) {
            neuralNetwork[i] = new Layer(width[i]);
        }

        //Input layer is not considered to be one
        for (int i = 0; i <= depth + 1; i++) {
            for (int k = 0; k < width[i]; k++) {
                neuralNetwork[i].setNode(k, learningRate, activationFunction, basisFunctions);
            }
        }
    }

    @Override
    protected Matrix generateW0(int n) {
        return LinearAlgebra.randMatrix(n, 1, MAX_RAND, MIN_RAND);
    }

    //Assume each column contains a feature and each row is a sample
    //Pass a
    @Override
    protected Matrix predictProtected(Matrix X) {
        Matrix y = new Matrix(X.getRows(), 1);

        for (int i = 1; i <= X.getRows(); i ++) {
            y.setValue(i, 1, predictSample(LinearAlgebra.vectorFromRow(X, i)));
        }

        return y;
    }

    private double predictSample(Matrix inputs) {
        Matrix x = inputs;
        for (int i = 0; i < neuralNetwork.length; i++) {
            x = neuralNetwork[i].fire(x);
        }

        double max = 0.0;
        for (int i = 1; i <= x.getRows(); i++) {
            if (Math.abs(x.getValue(i, 1)) > Math.abs(max)) {
                max = Math.abs(x.getValue(i, 1));
            }
        }

        return max;
    }

    @Override
    protected void trainProtected(Matrix x, Matrix y, Matrix w0) {
    }
}
