//Logistic Regression Classifier

import java.util.function.Function;

public class LogRegClassifier extends Model{

    public LogRegClassifier(double learningRate, Function<Double, Double>[] basisFunctions) {
        super(learningRate, basisFunctions);
    }

    @Override
    protected Matrix predictProtected(Matrix x) {
        Matrix y = buildFunction(x);

        //Only works for binary classification
        return LinearAlgebra.applyFunction(y, sigmoid);
    }

    @Override
    protected void trainProtected(Matrix x, Matrix y) {
        Matrix w0 = LinearAlgebra.zeroMatrix(x.getCols() * (basisFunctions.length - 1) + 1, 1);
        weights = Regression.logisticReg(x, y, w0, learningRate, basisFunctions, false);
    }

}
