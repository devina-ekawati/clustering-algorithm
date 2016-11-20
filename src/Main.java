import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * Created by Devina Ekawati on 11/19/2016.
 */
public class Main {
    public Instances loadData(String filename) {
        ConverterUtils.DataSource source;
        Instances data = null;
        try {
            source = new ConverterUtils.DataSource(filename);
            data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }

    public static void main(String args[]) {
        Main m = new Main();

        Instances data = m.loadData("data/weather.numeric.arff");

        MyKMeans kmeans = new MyKMeans(2);

        try {
            kmeans.buildClusterer(data);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
