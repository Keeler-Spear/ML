import java.util.function.Function;

public class ActivationFunctions {

    public static Function<Double, Double> sigmoid = (z) -> 1 / (1 + Math.exp(-z));

    //ToDo: Softmax

}
