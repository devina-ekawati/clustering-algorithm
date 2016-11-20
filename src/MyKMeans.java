import weka.clusterers.AbstractClusterer;
import weka.clusterers.Clusterer;
import weka.core.Capabilities;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

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
        HashMap<Integer, ArrayList<Instance>> cluster = new HashMap<>();
        EuclideanDistance euclideanDistance = new EuclideanDistance(data);
        ArrayList<Integer> initialSeeds = initSeeds(data);

        // Inisialisasi
        for (Integer seed : initialSeeds) {
            cluster.put(seed, new ArrayList<>());
        }

        // Assign instance ke cluster dgn euclidean distance
        for (int i = 0; i < data.numInstances(); i++) {
            System.out.println("-------Instance ke " + i + " -------");

            System.out.print("value sparse: ");
            for (int j = 0; j < data.instance(i).numAttributes(); j++) {
                System.out.print(data.instance(i).value(j) + " ");
            }
            System.out.println();

            if (!initialSeeds.contains(i)) {
                ArrayList<Double> distances = new ArrayList<>();
                for (Integer seed : initialSeeds) {
                    distances.add(euclideanDistance.distance(data.instance(seed), data.instance(i)));
                }

                System.out.println("Distance");
                for (Double distance : distances) {
                    System.out.print(distance + " ");
                }
                System.out.println("Cluster: " + initialSeeds.get(distances.indexOf(Collections.min(distances))));

                int min = initialSeeds.get(distances.indexOf(Collections.min(distances)));
                ArrayList<Instance> value = cluster.get(min);
                value.add(data.instance(i));

                cluster.put(min, value);
            }
        }

        for (Integer key : cluster.keySet()) {
            System.out.println("Cluster: " + key);
            for (Instance value : cluster.get(key)) {
                System.out.println("\t" + value + " ");
            }
        }
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
