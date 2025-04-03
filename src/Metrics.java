import java.util.HashSet;
import java.util.Set;
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

    private static final Matrix THRESHOLDS = LinearAlgebra.linSpace(0.0, 1.0, 0.01);

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

        int numClass = countClasses(exact);

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
     * @param xTest The x values of the exact data.
     * @param yTest The true data set.
     * @param w The weights of the function from which an approximation will be created.
     * @param fnc The functions used to model the data.
     * @return The confusion matrix based on the data sets provided.
     * @throws IllegalArgumentException If the data sets provided have more than one column.
     * @throws IllegalArgumentException If the data sets provided are not the same length.
     */
    public static Matrix confusionMatrix(Matrix xTest, Matrix yTest, Matrix w, Function[] fnc) {
        if (yTest.getCols() != 1 ) {
            throw new IllegalArgumentException("The exact data set must have exactly one column!");
        }

        if (yTest.getRows() != xTest.getRows()) {
            throw new IllegalArgumentException("The data sets must be the same length!");
        }

        Matrix approx = Regression.buildLogisticFunction(xTest, w, fnc);

        return confusionMatrix(yTest, approx);
    }

    //Counts the number of classes.
    private static int countClasses(Matrix y) {
        int n = 0;
        y = LinearAlgebra.roundMatrix(y);
        Set<Integer> classes = new HashSet<>();

        for (int i = 1; i <= y.getRows(); i++) {
            if (!classes.contains((int) y.getValue(i, 1))) {
                classes.add((int) y.getValue(i, 1));
                n++;
            }
        }

        return n;
    }

    //The probability that an object is correctly classified.
    //ToDo: Should be diag sum divided by all values.
    public static double accuracy(Matrix CM) {
        return LinearAlgebra.diagSum(CM) / LinearAlgebra.matrixSum(CM);
    }

    //The probability that a predicted positive is actually a positive.
    public static double precision(Matrix CM) {
        return CM.getValue(2, 2) / (CM.getValue(1, 2) + CM.getValue(2, 2));
    }

    //The probability that an actual positive was identified as such.
    public static double recall(Matrix CM) {
        return CM.getValue(2, 2) / (CM.getValue(2, 1) + CM.getValue(2, 2));
    }

    //Harmonic mean of precision and recall
    public static double fMeasure(Matrix CM) {
        double r = recall(CM);
        double p = precision(CM);
        return (2 * r * p) / (r + p);
    }

    //ToDo: Generalize for multi-classification
    //https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    public static void printClassificationReport (Matrix xTest, Matrix yTest, Matrix w, Function[] fnc) {
        Matrix CM = confusionMatrix(xTest, yTest, w, fnc);

        if (CM.getRows() != 2 && CM.getCols() != 2) {
            throw new IllegalArgumentException("The confusion matrix must be 2x2!");
        }
        System.out.println("Classification Report\n" + "---------------------");
        //Printing the confusion matrix
        System.out.println("Confusion Matrix:");
        for (int i = 0; i < CM.getRows(); i++) {
            System.out.print("Actual " + i + " - " + LinearAlgebra.transpose(LinearAlgebra.vectorFromRow(CM, i + 1)) + "\n");
        }
        System.out.print("Predicted:     ");
        for (int i = 0; i < CM.getCols(); i++) {
            System.out.print(i + "      ");
        }

        //General Metrics
        System.out.println("\n\nAccuracy: " + accuracy(CM));
        System.out.println("Precision: " + precision(CM));
        System.out.println("Recall: " + recall(CM));

        double p;
        double r;
        double f;
        String pString;
        String rString;
        String fString;
        String sString;
        double[] ps = new double[CM.getRows()];
        double[] rs = new double[CM.getCols()];
        double[] fs = new double[CM.getCols()];

        //Class specific metrics
        System.out.println("\n\t\t\tPrecision\tRecall\tF1-score\tSupport");
        for (int i = 0; i < CM.getRows(); i++) {
            p = CM.getValue(i + 1, i + 1) / LinearAlgebra.colSum(CM, i + 1);
            ps[i] = p;
            r = CM.getValue(i + 1, i + 1) / LinearAlgebra.rowSum(CM, i + 1);
            rs[i] = r;
            f =  (2 * r * p) / (r + p);
            fs[i] = f;
            pString = String.format("%.2f", p);
            rString = String.format("%.2f", r);
            fString = String.format("%.2f", f);
            sString = String.format("%.0f", LinearAlgebra.rowSum(CM, i + 1));

            System.out.print("\t\t" + i + "     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);
            System.out.println();

        }

        sString = String.format("%.0f", LinearAlgebra.matrixSum(CM));
        String acc = String.format("%.2f", accuracy(CM));
        System.out.println(" Accuracy\t\t\t\t\t\t  " + acc + "\t\t   " + sString);
        pString = String.format("%.2f", Stat.mean(ps));
        rString = String.format("%.2f", Stat.mean(rs));
        fString = String.format("%.2f", Stat.mean(fs));
        System.out.println("Macro Avg     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);

        double[] support = new double[CM.getRows()];
        for (int i = 0; i < CM.getRows(); i++) {
            support[i] = LinearAlgebra.rowSum(CM, i + 1);
        }
        pString = String.format("%.2f", Stat.weightedMean(ps, support));
        rString = String.format("%.2f", Stat.weightedMean(rs, support));
        fString = String.format("%.2f", Stat.weightedMean(fs, support));
        System.out.println("Micro Avg     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);

        Matrix roc = getROCCurve(xTest, yTest, w, fnc);
        double auc = getAUC(LinearAlgebra.vectorFromColumn(roc, 2), LinearAlgebra.vectorFromColumn(roc, 1));
        String curve = "ROC Curve (AUC = " + String.format("%.2f", auc) + ")";
        PyChart.plot(LinearAlgebra.vectorFromColumn(roc, 1), LinearAlgebra.vectorFromColumn(roc, 2), curve, "False Positive Rate", "True Positive Rate", "Receiver Operating Characteristic");
    }

    public static void printClassificationReport (Matrix yExact, Matrix yApprox) {
        Matrix CM = confusionMatrix(yExact, yApprox);

        if (CM.getRows() != 2 && CM.getCols() != 2) {
            throw new IllegalArgumentException("The confusion matrix must be 2x2!");
        }
        System.out.println("Classification Report\n" + "---------------------");
        //Printing the confusion matrix
        System.out.println("Confusion Matrix:");
        for (int i = 0; i < CM.getRows(); i++) {
            System.out.print("Actual " + i + " - " + LinearAlgebra.transpose(LinearAlgebra.vectorFromRow(CM, i + 1)) + "\n");
        }
        System.out.print("Predicted:     ");
        for (int i = 0; i < CM.getCols(); i++) {
            System.out.print(i + "      ");
        }

        //General Metrics
        System.out.println("\n\nAccuracy: " + accuracy(CM));
        System.out.println("Precision: " + precision(CM));
        System.out.println("Recall: " + recall(CM));

        double p;
        double r;
        double f;
        String pString;
        String rString;
        String fString;
        String sString;
        double[] ps = new double[CM.getRows()];
        double[] rs = new double[CM.getCols()];
        double[] fs = new double[CM.getCols()];

        //Class specific metrics
        System.out.println("\n\t\t\tPrecision\tRecall\tF1-score\tSupport");
        for (int i = 0; i < CM.getRows(); i++) {
            p = CM.getValue(i + 1, i + 1) / LinearAlgebra.colSum(CM, i + 1);
            ps[i] = p;
            r = CM.getValue(i + 1, i + 1) / LinearAlgebra.rowSum(CM, i + 1);
            rs[i] = r;
            f =  (2 * r * p) / (r + p);
            fs[i] = f;
            pString = String.format("%.2f", p);
            rString = String.format("%.2f", r);
            fString = String.format("%.2f", f);
            sString = String.format("%.0f", LinearAlgebra.rowSum(CM, i + 1));

            System.out.print("\t\t" + i + "     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);
            System.out.println();

        }

        sString = String.format("%.0f", LinearAlgebra.matrixSum(CM));
        String acc = String.format("%.2f", accuracy(CM));
        System.out.println(" Accuracy\t\t\t\t\t\t  " + acc + "\t\t   " + sString);
        pString = String.format("%.2f", Stat.mean(ps));
        rString = String.format("%.2f", Stat.mean(rs));
        fString = String.format("%.2f", Stat.mean(fs));
        System.out.println("Macro Avg     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);

        double[] support = new double[CM.getRows()];
        for (int i = 0; i < CM.getRows(); i++) {
            support[i] = LinearAlgebra.rowSum(CM, i + 1);
        }
        pString = String.format("%.2f", Stat.weightedMean(ps, support));
        rString = String.format("%.2f", Stat.weightedMean(rs, support));
        fString = String.format("%.2f", Stat.weightedMean(fs, support));
        System.out.println("Micro Avg     " + pString + "\t\t " + rString + "\t  " + fString + "\t\t   " + sString);
    }

    /**
     * Plots the ROC Curve which compares (True Positive Rate) to the FPR (False Positive Rate).
     * <P>
     *     This method was made by ChatGPT.
     * </P>
     *
     * @param xTest The x values of the exact data.
     * @param yTest The true data set.
     * @param w The weights of the function from which an approximation will be created.
     * @param fncs The functions used to model the data.
     */
    public static Matrix getROCCurve(Matrix xTest, Matrix yTest, Matrix w, Function[] fncs) {
        int numThresholds = THRESHOLDS.getRows();
        Matrix roc = new Matrix(numThresholds, 2);
        Matrix approx = Regression.buildLogisticFunction(xTest, w, fncs);

        for (int t = 1; t <= numThresholds; t++) {
            double threshold = THRESHOLDS.getValue(t, 1);

            // Build confusion matrix
            double[][] CM = new double[2][2];
            for (int i = 1; i <= yTest.getRows(); i++) {
                int actual = (int) yTest.getValue(i, 1);
                int predicted = round(approx.getValue(i, 1), threshold);
                CM[actual][predicted] += 1;
            }

            double tpr = CM[1][1] / (CM[1][1] + CM[1][0]);
            double fpr = CM[0][1] / (CM[0][1] + CM[0][0]);

            roc.setValue(t, 1, fpr);
            roc.setValue(t, 2, tpr);
        }

        return roc;
    }

    public static double[] quickModelEval (Matrix xTest, Matrix yTest, Matrix w, Function[] fnc) {
        double[] metrics = new double[2];
        Matrix roc = getROCCurve(xTest, yTest, w, fnc);
        metrics[0] = accuracy(confusionMatrix(xTest, yTest, w, fnc));
        metrics[1] = getAUC(LinearAlgebra.vectorFromColumn(roc, 2), LinearAlgebra.vectorFromColumn(roc, 1));
        System.out.println("Accuracy: " + String.format("%.4f", metrics[0]) + " | AUC: " + String.format("%.4f", metrics[1]));
        return metrics;
    }

    //This method was made by ChatGPT
    private static int round(double value, double threshold) {
        return value >= threshold ? 1 : 0;
    }

    //This method was made by ChatGPT. While I could use my own numerical integration method, those require a constant step size
    private static double getAUC(Matrix TPR, Matrix FPR) {
        double auc = 0.0;
        int n = TPR.getRows();

        for (int i = 1; i < n; i++) {
            double width = FPR.getValue(i, 1) - FPR.getValue(i + 1, 1);
            double height = (TPR.getValue(i, 1) + TPR.getValue(i + 1, 1)) / 2.0;
            auc += width * height;
        }

        return auc;
    }

}
