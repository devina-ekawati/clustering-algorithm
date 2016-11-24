import weka.clusterers.AbstractClusterer;
import weka.core.*;

import java.util.*;

/**
 * Created by Devina Ekawati on 11/19/2016.
 */
public class MyKMeans extends AbstractClusterer {
    private int k;
    private HashMap<Instance, ArrayList<Integer>> cluster;
    private EuclideanDistance euclideanDistance;
    private Instances instances;

    public MyKMeans(int k) {
        this.k = k;
    }

    private ArrayList<Integer> initSeeds(Instances data) {
        Random random = new Random();
        ArrayList<Integer> seeds = new ArrayList<>();

//        System.out.print("Initial seeds: ");
        while (seeds.size() < k) {
            Integer next;
            do {
                next = random.nextInt(data.numInstances());
            } while (seeds.contains(next));

            seeds.add(next);
        }
//        for (Integer seed : seeds) {
//            System.out.print(seed + " ");
//        }
//        System.out.println();

        return seeds;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        this.instances = data;
        cluster = new HashMap<>();

        ArrayList<Instance> seeds = new ArrayList<>();
        ArrayList<Integer> intialSeedsIdx = initSeeds(data);
        euclideanDistance = new EuclideanDistance(data);


        for (int i = 0; i < intialSeedsIdx.size(); i++) {
            seeds.add(data.instance(intialSeedsIdx.get(i)));
        }

        // Inisialisasi
        for (Instance seed : seeds) {
            cluster.put(seed, new ArrayList<>());
        }

        cluster = clusterInstances(data, seeds, cluster);

        boolean cek = true;
        int iteration = 0;

        do {
            iteration++;
            HashMap<Instance, ArrayList<Integer>> newCluster = new HashMap<>();
            ArrayList<Instance> newSeeds = new ArrayList<>();

            for (Instance key : cluster.keySet()) {
//                System.out.println("Cluster: " + key);
//                for (Integer value : cluster.get(key)) {
//                    System.out.println("\t" + value + " ");
//                }

                Instance newInstance = new DenseInstance(data.numAttributes());

                for (int i = 0; i < data.numAttributes(); i++) {
                    if (data.attribute(i).isNominal()) {
//                        System.out.println(data.attribute(i).value((int)getClusterMean(data,cluster.get(key),i,data.attribute(i).isNominal())));
                        newInstance.setValue(data.attribute(i), data.attribute(i).value((int)getClusterMean(data,cluster.get(key),i,data.attribute(i).isNominal())));
                    } else {
                        newInstance.setValue(data.attribute(i), getClusterMean(data,cluster.get(key),i,data.attribute(i).isNominal()));
                    }
                }

//                System.out.println(newInstance);

                newSeeds.add(newInstance);
            }

            for (Instance seed : newSeeds) {
                newCluster.put(seed, new ArrayList<>());
            }

            newCluster = clusterInstances(data,newSeeds,newCluster);

//            System.out.println();
//            System.out.println();
//            for (Instance key : newCluster.keySet()) {
//                System.out.println("New Cluster: " + key);
//                for (Integer value : newCluster.get(key)) {
//                    System.out.println("\t" + value + " ");
//                }
//            }

            if (isClustersEquals(cluster, newCluster) || iteration == 500) {
                cluster = new HashMap<>(newCluster);
                cek = false;
            }
        } while (cek);
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        double minDistance = Double.MAX_VALUE;
        int clusterID = 0;
        int minClusterID = clusterID;

        for (Instance key : cluster.keySet()) {
            for (Integer value : cluster.get(key)) {
                double newDistance = euclideanDistance.distance(instance, instances.instance(value));
                if (newDistance < minDistance) {
                    minClusterID = clusterID;
                    minDistance = newDistance;
                }
            }
            clusterID++;
        }
        return minClusterID;
    }

    public String toString() {
        StringBuffer stringBuffer = new StringBuffer();
        for (Instance key : cluster.keySet()) {
            stringBuffer.append("Centroid: " + key + "\n");
            for (Integer value : cluster.get(key)) {
                stringBuffer.append("\t" + instances.instance(value) + "\n");
            }
        }
        return stringBuffer.toString();
    }

    private HashMap<Instance, ArrayList<Integer>> clusterInstances(Instances data, ArrayList<Instance> seeds, HashMap<Instance, ArrayList<Integer>> newCluster) {
        for (int i = 0; i < data.numInstances(); i++) {
//                System.out.println("-------Instance ke " + i + " -------");

            ArrayList<Double> distances = new ArrayList<>();
            for (Instance seed : seeds) {
                distances.add(euclideanDistance.distance(seed, data.instance(i)));
            }

//                System.out.println("Distance");
//                for (Double distance : distances) {
//                    System.out.print(distance + " ");
//                }
//                System.out.println("New Cluster: " + newSeeds.get(distances.indexOf(Collections.min(distances))));

            Instance min = seeds.get(distances.indexOf(Collections.min(distances)));
            ArrayList<Integer> value = newCluster.get(min);
            value.add(i);

            newCluster.put(min, value);
        }

        return newCluster;
    }

    private boolean isClustersEquals(HashMap<Instance,ArrayList<Integer>> cluster, HashMap<Instance,ArrayList<Integer>> newCluster) {

        Instance[] key1 = cluster.keySet().toArray(new Instance[cluster.keySet().size()]);
        Instance[] key2 = newCluster.keySet().toArray(new Instance[newCluster.keySet().size()]);
        boolean isEquals = true;

        for (int i = 0; i < k; i++) {
            ArrayList<Integer> values1 = cluster.get(key1[i]);
            ArrayList<Integer> values2 = newCluster.get(key2[i]);

            if (values1.size() != values2.size()) {
                isEquals = false;
                break;
            } else {
                for (int j = 0; j < values1.size(); j++) {
                    if (values1.get(j) != values2.get(j)) {
                        isEquals = false;
                        break;
                    }
                }
                if (!isEquals) {
                    break;
                }
            }
        }
        return isEquals;
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    private double getClusterMean(Instances data, ArrayList<Integer> index, int clusterIndex, boolean isNominal) {
        double result;
        if (isNominal) {
            int numAttributeValues = data.get(0).attribute(clusterIndex).numValues();
            int[] countAttributes = new int[numAttributeValues];
            for (int i = 0; i < index.size(); i++) {
                countAttributes[(int)(data.instance(index.get(i)).value(clusterIndex))]++;
            }

            int idxMax = 0;
            for (int i = 1; i < numAttributeValues; i++) {
                if (countAttributes[idxMax] < countAttributes[i]) {
                    idxMax = i;
                }
            }
            result = (double) idxMax;

        } else {
            double sum = 0;
            for (int i = 0; i < index.size(); i++) {
                sum += data.instance(i).value(clusterIndex);
            }


            result = sum/(double)index.size();
        }
        return result;
    }
}
