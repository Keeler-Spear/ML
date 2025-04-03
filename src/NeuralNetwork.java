import java.util.function.Function;

public class NeuralNetwork extends Model{
    private static final double MAX_RAND = 10;
    private static final double MIN_RAND = 10;

    //Depth = number of hiden layers
    //width = number of nodes per hidden layer
    public NeuralNetwork(int depth, int[] width, String activationFunction, String objectiveFunction, double learningRate, Function<Double, Double>[] basisFunctions) {
        super(learningRate, basisFunctions);
    }

    @Override
    protected Matrix generateW0(int n) {
        return LinearAlgebra.randMatrix(n, 1, MAX_RAND, MIN_RAND);
    }

    @Override
    protected Matrix predictProtected(Matrix x) {
        return null;
    }

    @Override
    protected void trainProtected(Matrix x, Matrix y, Matrix w0) {
    }
}
