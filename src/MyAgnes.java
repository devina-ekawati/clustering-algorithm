import weka.classifiers.evaluation.Evaluation;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.pmml.Array;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Tifani on 11/19/2016.
 */
public class MyAgnes extends AbstractClusterer {
    /** link types */
    final static int SINGLE = 0;
    final static int COMPLETE = 1;
    final static int NUM_CLUSTER = 2;

    int linkType = SINGLE;
    ArrayList<ArrayList<ArrayList<Instance>>> clustersHierarchy = new ArrayList<>();
    HashMap<Instance, HashMap<Instance, Double>> distanceMatrix;

    public int getLinkType() {
        return linkType;
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        ArrayList<ArrayList<Instance>> clusters = new ArrayList<>();

        // Initialize cluster and distance matrix
        for(Instance instance : instances) {
            clusters.add(new ArrayList<>(Arrays.asList(instance)));
        }

        distanceMatrix = calculateDistanceMatrix(instances);
        joinNeighbor(clusters);
    }

    private HashMap<Instance, HashMap<Instance, Double>> calculateDistanceMatrix(Instances instances) {
        System.out.println("Calculate distance matrix");
        HashMap<Instance, HashMap<Instance, Double>> distanceMatrix = new HashMap<>();
        for(int i=0; i<instances.size(); i++) {
            Instance instanceA = instances.get(i);
            distanceMatrix.put(instanceA, new HashMap<>());
            for(int j=0; j<i; j++) {
                Instance instanceB = instances.get(j);
                HashMap<Instance, Double> neighbors = distanceMatrix.get(instanceA);
                neighbors.put(instanceB, distanceMatrix.get(instanceB).get(instanceA));
                distanceMatrix.put(instanceA, neighbors);
            }
            for(int j=i; j<instances.size(); j++) {
                Instance instanceB = instances.get(j);
                double distance = calculateDistance(instanceA, instanceB);
                HashMap<Instance, Double> neighbors = distanceMatrix.get(instanceA);
                neighbors.put(instanceB, distance);
                distanceMatrix.put(instanceA, neighbors);
            }
        }
        return distanceMatrix;
    }

    private double calculateDistance(Instance instanceA, Instance instanceB) {
        // System.out.println("Calculate distance: " + instanceA.toString() + " - " + instanceB.toString());
        double diff = 0;
        for(int i=0; i<instanceA.numAttributes(); i++) {
            if (instanceA.value(i) != instanceB.value(i)) {
                diff = diff + Math.pow(1, 2); // TODO: numerik
            }
        }
        return Math.sqrt(diff);
    }

    private void joinNeighbor(ArrayList<ArrayList<Instance>> clusters) {
        // System.out.println("Join neighbor");
        clustersHierarchy.add(clusters);

        if(clusters.size() > 1) {
            int clusterAMin = 0;
            int clusterBMin = 1;
            double minDistance = calculateClusterDistance(clusters.get(clusterAMin), clusters.get(clusterBMin));

            for(int i=0; i<clusters.size()-1; i++) {
                for (int j=i+1; j<clusters.size(); j++) {
                    double newDistance = calculateClusterDistance(clusters.get(i), clusters.get(j));
                    if (newDistance < minDistance) {
                        minDistance = newDistance;
                        clusterAMin = i;
                        clusterBMin = j;
                    }
                }
            }

            // join cluster
            ArrayList<ArrayList<Instance>> upperClusters = new ArrayList<>();
            for(int i=0; i<clusters.size(); i++) {
                if (i == clusterBMin) {
                    ArrayList newCluster = upperClusters.get(clusterAMin);
                    for(Instance instance : clusters.get(i)) {
                        newCluster.add(instance);
                    }
                } else {
                    ArrayList newCluster = new ArrayList();
                    for(Instance instance : clusters.get(i)) {
                        newCluster.add(instance);
                    }
                    upperClusters.add(newCluster);
                }
            }

            joinNeighbor(upperClusters);
        } else {
            ArrayList<ArrayList<Instance>> cluster2 = null;
            try {
                cluster2 = clustersHierarchy.get(clustersHierarchy.size()-numberOfClusters());
            } catch (Exception e) {
                e.printStackTrace();
            }

            for(int j=0; j<cluster2.size(); j++) {
                ArrayList<Instance> currentCluster = cluster2.get(j);
                System.out.println("Cluster-" + j);
                for(int k=0; k<currentCluster.size(); k++) {
                    System.out.println("    " + currentCluster.get(k).toString());
                }
            }
        }
    }

    public void printHierarchy() throws IOException {
        FileWriter fw = new FileWriter("result.txt");
        PrintWriter pw = new PrintWriter(fw);
        for(int i=0; i<clustersHierarchy.size(); i++) {
            System.out.println("Hierarchy: " + i);
            pw.println("Hierarchy: " + i);
            for(int j=0; j<clustersHierarchy.get(i).size(); j++) {
                ArrayList<Instance> currentCluster = clustersHierarchy.get(i).get(j);
                System.out.println("Cluster-" + j);
                pw.println("Cluster-" + j);
                for(int k=0; k<currentCluster.size(); k++) {
                    System.out.println("    " + currentCluster.get(k).toString());
                    pw.println("    " + currentCluster.get(k).toString());
                }
            }
            System.out.println();
            pw.println();
        }
        pw.flush();
        pw.close();
        fw.close();
    }

    private double calculateClusterDistance(ArrayList<Instance> clusterA, ArrayList<Instance> clusterB) {
        // System.out.println("Calculate cluster distance");
        double distance = distanceMatrix.get(clusterA.get(0)).get(clusterB.get(0));
        for(int i=0; i<clusterA.size(); i++) {
            for(int j=0; j<clusterB.size(); j++) {
                double newDistance = distanceMatrix.get(clusterA.get(i)).get(clusterB.get(j));
                if (linkType == SINGLE && newDistance < distance) {
                    distance = newDistance;
                } else if (linkType == COMPLETE && newDistance > distance) {
                    distance = newDistance;
                }
            }
        }
        return  distance;
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        if (clustersHierarchy.size() == 0) {
            return 0;
        } else {
            ArrayList<ArrayList<Instance>> clusters = clustersHierarchy.get(clustersHierarchy.size()-numberOfClusters());
            ArrayList<Instance> newCluster = new ArrayList<>();
            newCluster.add(instance);
            double distance = calculateClusterDistance(newCluster, clusters.get(0));
            int cluster = 0;
            for(int i=0; i<clusters.size(); i++) {
                double newDistance = calculateClusterDistance(newCluster, clusters.get(i));
                if (newDistance < distance) {
                    distance = newDistance;
                    cluster = i;
                }
            }
            return cluster;
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (numberOfClusters() == 0) {
            double[] p = new double[1];
            p[0] = 1;
            return p;
        }
        double[] p = new double[numberOfClusters()];
        p[clusterInstance(instance)] = 1.0;
        return p;
    }

    @Override
    public int numberOfClusters() throws Exception {
        return NUM_CLUSTER;
    }

    public static void main(String[] args) throws Exception {
        Instances data = loadData("data/iris.arff");
        MyAgnes myAgnes = new MyAgnes();

        ClusterEvaluation eval = new ClusterEvaluation();
        Clusterer clusterer = AbstractClusterer.makeCopy(myAgnes);

        eval.setClusterer(clusterer); // TODO: di sini sininya kayanya masih kobam, either si myagnesnya ga kebuild hierarchynya atau classify instancenya error @_@
        myAgnes.buildClusterer(data);
        eval.evaluateClusterer(data);

        System.out.println("NUMCLUSTER: " + eval.getNumClusters());

        System.out.println();
        System.out.println();

        System.out.println("Results");
        System.out.println("=======");
        System.out.println(eval.clusterResultsToString());

    }

    public static Instances loadData(String filename) {
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
}
