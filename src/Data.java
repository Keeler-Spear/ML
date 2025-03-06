/**
 * A static class that provides methods manipulating data.
 * <p>
 *     This class operations such as feature scaling, bad data handling, and more.
 * </p>
 *
 * @author Keeler Spear
 * @version %I%, %G%
 * @since 1.0
 */
public class Data {

    final static double TOL = 0.000001;

    //Computes if the provided value is "zero."
    private static boolean isZero(double val) {
        if (Math.abs(val) < TOL) {
            return true;
        }
        else {
            return false;
        }
    }

    /**
     * Creates binary classification data sets for model training and evaluation.
     *
     * @param file The name of the csv file containing the raw data.
     * @param class1 The string representation of the first class.
     * @param class2 The string representation of the second class.
     * @param skip The number of columns that should be skipped on the left before data collection begins.
     * @param split The percentage of data to be used for model training. 100% - split will be the percent of data used
     *              for model testing.
     * @param labelAtStart If the samples' label appears in the first column after those being skipped. If the
     *              parameter is false, the data will be treated as if its last column contains the labels.
     * @param clean If the features with the value of 0 should be set to the median value of that feature.
     * @param scale If the features should be scaled.
     * @return An array of matrices containing:
     *         <ul>
     *             <li> The sample parameters to be used for training</li>
     *             <li> The sample labels to be used for training</li>
     *             <li> The sample parameters to be used for testing</li>
     *             <li> The sample labels to be used for testing</li>
     *         </ul>
     * @throws IllegalArgumentException If split is not a percentage.
     */
    public static Matrix[] getBCData(String file, String class1, String class2, int skip, double split, boolean labelAtStart, boolean clean, boolean scale) {
        if (split < 0 || split > 100) {
            throw new IllegalArgumentException("Split must be between 0 and 100!");
        }

        double[][] dataVals = ReadFile.csvToArray(file, class1, class2, skip);

        if (clean) {
            Matrix temp = new Matrix(dataVals);

            clean(temp, labelAtStart);

            dataVals = temp.getMatrix();

        }

        if (scale) {
            scaleFeatures(dataVals, labelAtStart);
        }

        int mid = (int) (dataVals.length * (split / 100.0));

        double[][] trData = new double[mid][dataVals[0].length];
        double[][] teData = new double[dataVals.length - mid][dataVals[0].length];

        //Splitting data
        for (int i = 0; i < trData.length; i++) {
            trData[i] = dataVals[i];
        }

        for (int i = 0; i < teData.length; i++) {
            teData[i] = dataVals[i + trData.length];
        }

        int labelCol;

        if (labelAtStart) {
            labelCol = 1;
        }
        else {
            labelCol = dataVals[0].length;
        }

        Matrix xTrain = new Matrix(trData);
        Matrix yTrain = LinearAlgebra.vectorFromColumn(xTrain, labelCol);
        xTrain.removeCol(labelCol); //Removing labels

        Matrix xTest = new Matrix(teData);
        Matrix yTest = LinearAlgebra.vectorFromColumn(xTest, labelCol);
        xTest.removeCol(labelCol); //Removing labels

        return new Matrix[]{xTrain, yTrain, xTest, yTest};
    }

    //Sets the features with the value of 0 to the median value of that feature.
    private static void clean(Matrix data, boolean labelAtStart) {
        double mean;
        int frontOffset = 0;
        int backOffset = 0;

        if (labelAtStart) {
            frontOffset = 1;
        }
        else {
            backOffset = 1;
        }
        //Finding the median of the column
        for (int i = 1 + frontOffset; i <= data.getCols() - backOffset; i++) {
            mean = Stat.mean(data.getCol(i));
            for (int j = 1; j <= data.getRows(); j++) {
                if (isZero(data.getValue(j, i))) {
                    data.setValue(j, i, mean);
                }
            }
        }
    }

    private static void scaleFeatures(double[][] data, boolean labelAtStart) {
        double min;
        double max;
        int frontOffset = 0;
        int backOffset = 0;

        if (labelAtStart) {
            frontOffset = 1;
        }
        else {
            backOffset = 1;
        }

        for (int i = frontOffset; i < data[0].length - backOffset; i++) {
            min = 0.0;
            max = 0.0;
            //Finding max and min
            for (int j = 0; j < data.length; j++) {
                if (data[j][i] > max) {
                    max = data[j][i];
                }
                if (data[j][i] < min) {
                    min = data[j][i];
                }
            }

            //Scaling that feature's data
            for (int j = 0; j < data.length; j++) {
                data[j][i] = (data[j][i] - min) / (max - min);
            }
        }
    }

}
