import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;
import weka.core.Option;
import weka.core.converters.ConverterUtils;

import java.util.Scanner;

/**
 * Created by Devina Ekawati on 11/19/2016.
 */
public class Main {
    public static final String BASE_PATH = "data/";
    public static final String DATASET_BREASTCANCER = "breast-cancer.arff";
    public static final String DATASET_CONTACTLENNSES = "contact-lenses.arff";
    public static final String DATASET_IRIS = "iris.arff";
    public static final String DATASET_WEATHERNOMINAL = "weather.nominal.arff";
    public static final String DATASET_WEATHERNUMERIC = "weather.numeric.arff";
    public static final String DATASET_CRIMINAL = "criminal.arff";

    private static Instances loadData(String filename) {
        ConverterUtils.DataSource source;
        Instances data = null;
        try {
            source = new ConverterUtils.DataSource(filename);
            data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (data.classIndex() == -1)
//                data.setClassIndex(data.numAttributes() - 1);
                data.setClassIndex(0);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return data;
    }

    public static void testAgnesWeka(String datapath, int numCluster, int linkType) throws Exception {
        System.out.println("============= AGNES WEKA ===========");
        System.out.println();
        System.out.println("Data            : \"" + datapath + "\"");

        Instances data = loadData(datapath);

        ClusterEvaluation eval = new ClusterEvaluation();
        Clusterer clusterer = new HierarchicalClusterer();
        ((HierarchicalClusterer) clusterer).setNumClusters(numCluster);
        switch (linkType) {
            case MyAgnes.SINGLE:
                ((HierarchicalClusterer) clusterer).setOptions(new String[] {"-L", "SINGLE"});
                break;
            case MyAgnes.COMPLETE:
                ((HierarchicalClusterer) clusterer).setOptions(new String[] {"-L", "COMPLETE"});
                break;
        }
        clusterer.buildClusterer(data);

        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);

        System.out.println();

        System.out.println("Results");
        System.out.println("=======");
        System.out.println(eval.clusterResultsToString());
    }

    public static void testMyAgnes(String datapath, int numCluster, int linkType) throws Exception {
        System.out.println("=============== MY AGNES ===============");
        System.out.println();
        System.out.println("Data            : \"" + datapath + "\"");
        System.out.println("Jumlah cluster  : " + numCluster);
        System.out.print  ("Tipe link       : ");
        if (linkType == MyAgnes.SINGLE) System.out.println("Singe-link");
            else System.out.println("Complete-link");

        Instances data = loadData(datapath);

        ClusterEvaluation eval = new ClusterEvaluation();
        Clusterer clusterer = new MyAgnes(numCluster, linkType);
        clusterer.buildClusterer(data);

        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);

        System.out.println();

        System.out.println("Results");
        System.out.println("=======");
        System.out.println(eval.clusterResultsToString());
    }

    public static void testMyKMeans(String datapath, int k) throws Exception{
        System.out.println("============= MY K-MEANS ===========");
        System.out.println();
        System.out.println("Data: \"" + datapath + "\"");

        Instances data = loadData(datapath);

        ClusterEvaluation eval = new ClusterEvaluation();
        Clusterer clusterer = new MyKMeans(k);
        clusterer.buildClusterer(data);

        eval.setClusterer(clusterer);
        eval.evaluateClusterer(data);

        System.out.println();

        System.out.println("Results");
        System.out.println("=======");
        System.out.println(eval.clusterResultsToString());
    }

    public static void main(String args[]) {
        try {
//            Main.testMyAgnes("data/criminal.arff", 6, MyAgnes.SINGLE);
            Main.testMyKMeans("data/criminal.arff", 2);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
