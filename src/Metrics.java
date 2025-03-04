import java.util.function.Function;

/**
 * A static class that provides methods for evaluating a machine learning model.
 * <p>
 *     This class operations such as confusion matrices, classification reports, and more.
 * </p>
 *
 * @author Keeler Spear
 * @version %I%, %G%
 * @since 1.0
 */
public class Metrics {

    /**
     * Computes the mean squared error between two data sets.
     *
     * @param exact The true data set.
     * @param approx The approximation of the true data set.
     * @return The mean squared error between two data sets.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static double meanSquared(double[] exact, double[] approx) {
        if (exact.length != approx.length) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }
        int n = exact.length;
        double sum = 0.0;

        for (int i = 0; i < n; i++) {
            sum += Math.pow(exact[i] - approx[i], 2);
        }

        return sum / n;
    }

    /**
     * Computes the mean squared error between two data sets.
     *
     * @param exact The true data set.
     * @param approx The approximation of the true data set.
     * @return The mean squared error between two data sets.
     * @throws IllegalArgumentException If the data sets provided have more than one column.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static double meanSquared(Matrix exact, Matrix approx) {
        if (exact.getCols() != 1 || approx.getCols() != 1) {
            throw new IllegalArgumentException("The data sets must have one column each!");
        }

        if (exact.getRows() != approx.getRows()) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        int n = exact.getRows();
        double sum = 0.0;

        for (int i = 1; i <= n; i++) {
            sum += Math.pow(exact.getValue(i, 1) - approx.getValue(i, 1), 2);
        }

        return sum / n;
    }

    /**
     * Computes the mean squared error between two data sets, where the approximate data set will be created from a
     * function created by the weights provided.
     *
     * @param x The x values of the exact data.
     * @param exact The true data set.
     * @param w The weights of the function from which an approximation will be created.
     * @param fnc The functions used to model the data.
     * @return The mean squared error between two data sets.
     * @throws IllegalArgumentException If the exact data set provided has more than one column.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static double meanSquared(Matrix x, Matrix exact, Matrix w, Function[] fnc) {
        if (exact.getCols() != 1 ) {
            throw new IllegalArgumentException("The exact data set must have exactly one column!");
        }

        if (exact.getRows() != x.getRows()) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        Matrix approx = Regression.buildFunction(x, w, fnc);

        int n = exact.getRows();
        double sum = 0.0;

        for (int i = 1; i <= n; i++) {
            sum += Math.pow(exact.getValue(i, 1) - approx.getValue(i, 1), 2);
        }

        return sum / n;
    }

    //ToDo: r^2 or coefficient of determination

    /**
     * Creates a confusion matrix based on the data sets provided.
     * <p>
     *     DOES NOT WORK FOR MULTI-CLASSIFICATION!
     * </p>
     *
     * @param exact The true data set.
     * @param approx The approximation of the true data set.
     * @return The confusion matrix based on the data sets provided.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static Matrix confusionMatrix(int[] exact, int[] approx) {
        if (exact.length != approx.length) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        double[][] CM = new double[2][2];

        for (int i = 0; i < exact.length; i++) {
            if (exact[i] == 1 && approx[i] == 1) { //True positive
                CM[0][0] += 1;
            }
            else if (exact[i] == 0 && approx[i] == 1) { //False positive
                CM[0][1] += 1;
            }
            else if (exact[i] == 1 && approx[i] == 0) { //False negative
                CM[1][0] += 1;
            }
            else if (exact[i] == 0 && approx[i] == 0) { //True negative
                CM[1][1] += 1;
            }
        }

        return new Matrix(CM);
    }

    /**
     * Creates a confusion matrix based on the data sets provided.
     *
     * @param exact The true data set.
     * @param approx The approximation of the true data set.
     * @return The confusion matrix based on the data sets provided.
     * @throws IllegalArgumentException If the data sets provided have more than one column.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static Matrix confusionMatrix(Matrix exact, Matrix approx) {
        if (exact.getCols() != 1 ) {
            throw new IllegalArgumentException("The exact data set must have exactly one column!");
        }

        if (exact.getRows() != approx.getRows()) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        int numClass = Regression.countClasses(exact);

        double[][] CM = new double[numClass][numClass];

        //ASSUMES THE CLASSES ARE IN NUMERICAL ORDER, WITH 0 AS AN INCLUDED VALUE
        for (int i = 1; i <= exact.getRows(); i++) {
            CM[(int) Math.round(exact.getValue(i, 1))][(int) Math.round(approx.getValue(i, 1))] += 1;
        }


        return new Matrix(CM);
    }

    //Matrix x, Matrix exact, Matrix w, Function[] fnc

    /**
     * Creates a confusion matrix based from two data sets, where the approximate data set will be created from a
     * function created by the weights provided.
     *
     * @param x The x values of the exact data.
     * @param exact The true data set.
     * @param w The weights of the function from which an approximation will be created.
     * @param fnc The functions used to model the data.
     * @return The confusion matrix based on the data sets provided.
     * @throws IllegalArgumentException If the data sets provided have more than one column.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static Matrix confusionMatrix(Matrix x, Matrix exact, Matrix w, Function[] fnc) {
        if (exact.getCols() != 1 ) {
            throw new IllegalArgumentException("The exact data set must have exactly one column!");
        }

        if (exact.getRows() != x.getRows()) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        Matrix approx = Regression.buildLogisticFunction(x, w, fnc);

        return confusionMatrix(exact, approx);
    }

    //The probability that an object is correctly classified.
    public static double accuracy(Matrix CM) {
        return (CM.getValue(1, 1) + CM.getValue(2, 2)) / (CM.getValue(1, 1) + CM.getValue(1, 2) + CM.getValue(2, 1) + CM.getValue(2, 2));
    }

    //The probability that a predicted positive is actually a positive.
    public static double precision(Matrix CM) {
        return CM.getValue(2, 2) / (CM.getValue(1, 2) + CM.getValue(2, 2));
    }

    //The probability that an actual positive was identified as such.
    public static double recall(Matrix CM) {
        return CM.getValue(2, 2) / (CM.getValue(2, 1) + CM.getValue(2, 2));
    }

    //
    public static double fMeasure(Matrix CM) {
        double r = recall(CM);
        double p = precision(CM);
        return (2 * r * p) / (r + p);
    }

    //https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    public static void printClassificationReport (Matrix CM) {
        System.out.println("Classification Report\n" + "---------------------");
        System.out.println("Confusion Matrix:");
        System.out.println(CM);
        System.out.println("Accuracy: " + accuracy(CM));
        System.out.println("Precision: " + precision(CM));
        System.out.println("Recall: " + recall(CM));

    }

    //ToDo: ROC curve

}
