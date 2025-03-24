import java.util.Random;

public class Perceptron {
    private final double[] weights;
    private double bias;
    private final double learningRate;
    private final boolean useBias;
    

    public Perceptron(int inputSize, double learningRate, boolean useBias) {
        this.weights = new double[inputSize];
        this.learningRate = learningRate;
        this.useBias = useBias;
        
        Random random = new Random();
        for (int i = 0; i < inputSize; i++) {
            this.weights[i] = random.nextDouble() * 2 - 1;
        }
        
        if (useBias) {
            this.bias = random.nextDouble() * 2 - 1;
        } else {
            this.bias = 0;
        }
    }

    
    /**
     * Función de activación sigmoid: 1/(1 + e^(-x))
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
    La salida del perceptrón funcion sigmoide
     */
    public double calculateOutput(double[] inputs) {
        double sum = useBias ? bias : 0;
        
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sigmoid(sum);
    }
    
    /**
    1 si la salida es >= 0.5 o 0 escalonado?
     */
    public int predict(double[] inputs) {
        return calculateOutput(inputs) >= 0.5 ? 1 : 0;
    }
    
    /**
    etapa de entrenamiento
     */
    public void train(double[][] trainingData, double[] targets) {
        int maxEpochs = 100;
        double errorThreshold = 0.01;
        
        int epoch = 0;
        boolean converged = false;
        
        System.out.println("Iniciando entrenamiento del perceptron " +
                            (useBias ? "con sesgo: " : "sin sesgo: "));
        System.out.println("Pesos iniciales: " + weightsToString());
        if (useBias) {
            System.out.println("Sesgo inicial: " + String.format("%.4f", bias));
        }
        
        // Entrenamiento por épocas
        while (!converged && epoch < maxEpochs) {
            double totalError = 0;
            
            for (int i = 0; i < trainingData.length; i++) {
                double[] inputs = trainingData[i];
                double target = targets[i];
                
                // calcular salida actual
                double output = calculateOutput(inputs);
                
                // calcular error
                double error = target - output;
                totalError += Math.pow(error, 2); // Error cuadrático
                
                // Factor derivada de la función sigmoid: output * (1 - output)
                double sigmoidDerivative = output * (1 - output);
                
                // se actualizan los pesos
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * sigmoidDerivative * inputs[j];
                }
                
                // se actualiza el sesgo si se necesita
                if (useBias) {
                    bias += learningRate * error * sigmoidDerivative;
                }
            }
            
            // se calcula el error promedio
            double mse = totalError / trainingData.length;
            epoch++;
            
            // Converge?
            if (mse < errorThreshold) {
                converged = true;
            }
            
            // como se visualiza el el progreso de las epocas
            if (epoch % 100 == 0) {
                System.out.println("Epoca " + epoch + ", Error: " + mse);
            }
        }
        
        System.out.println("Entrenamiento completado en " + epoch + " epocas.");
        System.out.println("Pesos finales: " + weightsToString());
        if (useBias) {
            System.out.println("Sesgo final: " + String.format("%.4f", bias));
        }
        
        printResults(trainingData, targets);
    }
    
    /**
     * Convierte los pesos a una cadena para mostrarlos
     */
    private String weightsToString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < weights.length; i++) {
            sb.append(String.format("%.4f", weights[i]));
            if (i < weights.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
    

    public void printResults(double[][] data, double[] expectedOutputs) {
        System.out.println("\n--- Resultados detallados ---");
        
        for (int i = 0; i < data.length; i++) {
            double[] inputs = data[i];
            double expected = expectedOutputs[i];
            
            // suma ponderada
            double weightedSum = useBias ? bias : 0;
            for (int j = 0; j < weights.length; j++) {
                weightedSum += inputs[j] * weights[j];
            }
            
            // salida usando la función sigmoid
            double output = sigmoid(weightedSum);
            int prediction = output >= 0.5 ? 1 : 0;
            
            
            System.out.print("Entradas: [");
            for (int j = 0; j < inputs.length; j++) {
                System.out.print((int)inputs[j]); // Se convierte a entero
                if (j < inputs.length - 1) System.out.print(", ");
            }
            System.out.println("]");
            
            if (useBias) {
                System.out.println("Suma Ponderada = " + String.format("%.4f", weightedSum));
            } else {
                System.out.println("Suma Ponderada = " + String.format("%.4f", weightedSum));
            }
            
            System.out.println("Sigmoid = " + String.format("%.4f", output));
            System.out.println("Prediccion: " + prediction + ", Esperado: " + (int)expected);
            System.out.println("------------------------");
        }
    }
}