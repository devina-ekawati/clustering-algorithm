import weka.clusterers.AbstractClusterer;
import weka.core.*;

import java.util.*;

/**
 * Created by Devina Ekawati on 11/19/2016.
 */
public class MyKMeans extends AbstractClusterer {
    private int k;

    public MyKMeans(int k) {
        this.k = k;
    }

    private ArrayList<Integer> initSeeds(Instances data) {
        Random random = new Random();
        ArrayList<Integer> seeds = new ArrayList<>();
        System.out.print("Initial seeds: ");
        while (seeds.size() < k) {
            Integer next = random.nextInt(data.numInstances());
            seeds.add(next);
        }
        for (Integer seed : seeds) {
            System.out.print(seed + " ");
        }
        System.out.println();

        return seeds;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        HashMap<Instance, ArrayList<Instance>> cluster = new HashMap<>();
        EuclideanDistance euclideanDistance = new EuclideanDistance(data);
        ArrayList<Instance> seeds = new ArrayList<>();
        ArrayList<Integer> intialSeedsIdx = initSeeds(data);

        for (int i = 0; i < intialSeedsIdx.size(); i++) {
            seeds.add(data.instance(intialSeedsIdx.get(i)));
        }


        // Inisialisasi
        for (Instance seed : seeds) {
            cluster.put(seed, new ArrayList<>());
        }

        // Assign instance ke cluster dgn euclidean distance
        for (int i = 0; i < data.numInstances(); i++) {
            System.out.println("-------Instance ke " + i + " -------");

            ArrayList<Double> distances = new ArrayList<>();
            for (Instance seed : seeds) {
                distances.add(euclideanDistance.distance(seed, data.instance(i)));
            }

            System.out.println("Distance");
            for (Double distance : distances) {
                System.out.print(distance + " ");
            }
            System.out.println("Cluster: " + seeds.get(distances.indexOf(Collections.min(distances))));

            Instance min = seeds.get(distances.indexOf(Collections.min(distances)));
            ArrayList<Instance> value = cluster.get(min);
            value.add(data.instance(i));

            cluster.put(min, value);
        }

        for (Instance key : cluster.keySet()) {
            System.out.println("Cluster: " + key);
            for (Instance value : cluster.get(key)) {
                System.out.println("\t" + value + " ");
            }

            Instance newInstance = new DenseInstance(data.numAttributes());

            for (int i = 0; i < data.numAttributes(); i++) {
                newInstance.setValue(data.attribute(i), getClusterMean(cluster.get(key),i,data.attribute(i).isNominal()));
                System.out.println(getClusterMean(cluster.get(key),i,data.attribute(i).isNominal()));
            }

            System.out.println(newInstance);
        }

    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    private double getClusterMean(ArrayList<Instance> instances, int clusterIndex, boolean isNominal) {
        double result;
        if (isNominal) {
            int numAttributeValues = instances.get(0).attribute(clusterIndex).numValues();
            int[] countAttributes = new int[numAttributeValues];
            for (Instance instance : instances) {
                countAttributes[(int)(instance.value(clusterIndex))]++;
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
            for (Instance instance : instances) {
                sum += instance.value(clusterIndex);
            }


            result = sum/(double)instances.size();
        }
        return result;
    }
}
