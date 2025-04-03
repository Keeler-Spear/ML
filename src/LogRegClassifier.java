//Logistic Regression Classifier

import java.util.function.Function;

public class LogRegClassifier extends Model{

    public LogRegClassifier(double learningRate, Function<Double, Double>[] basisFunctions) {
        super(learningRate, basisFunctions);
    }

    @Override
    protected Matrix generateW0(int n) {
        return LinearAlgebra.zeroMatrix(n, 1);
    }

    @Override
    protected Matrix predictProtected(Matrix x) {
        Matrix y = buildFunction(x);

        //Only works for binary classification
        return LinearAlgebra.applyFunction(y, sigmoid);
    }

    @Override
    protected void trainProtected(Matrix X, Matrix y, Matrix w0) {
        weights = Regression.logisticReg(X, y, w0, learningRate, basisFunctions, false);
    }

}
