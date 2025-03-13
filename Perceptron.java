import java.util.Random;

/**
 * Clase que implementa un perceptrón simple, con o sin sesgo
 */
public class Perceptron {
    // Los pesos del perceptrón
    private final double[] weights;
    
    // El sesgo (bias) para el perceptrón con sesgo
    private double bias;
    
    // La tasa de aprendizaje
    private final double learningRate;
    
    // Indica si este perceptrón usa sesgo o no
    private final boolean useBias;
    
    /**
     * Constructor para crear un perceptrón
     * @param inputSize Número de entradas
     * @param learningRate Tasa de aprendizaje
     * @param useBias Si es true, crea un perceptrón con sesgo; si es false, sin sesgo
     */
    public Perceptron(int inputSize, double learningRate, boolean useBias) {
        this.weights = new double[inputSize];
        this.learningRate = learningRate;
        this.useBias = useBias;
        
        // Inicializar pesos y sesgo con valores aleatorios entre -1 y 1
        Random random = new Random();
        for (int i = 0; i < inputSize; i++) {
            this.weights[i] = random.nextDouble() * 2 - 1;
        }
        
        if (useBias) {
            this.bias = random.nextDouble() * 2 - 1;
        } else {
            this.bias = 0; // No se usa, pero inicializamos a 0
        }
    }
    
    /**
     * Función de activación sigmoid: 1/(1 + e^(-x))
     * @param x El valor de entrada
     * @return Un valor entre 0 y 1
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * Calcula la salida del perceptrón para un conjunto de entradas
     * @param inputs Vector de entradas
     * @return La salida del perceptrón (un valor entre 0 y 1)
     */
    public double calculateOutput(double[] inputs) {
        double sum = useBias ? bias : 0; // Usamos el sesgo solo si useBias es true
        
        // Calculamos la suma ponderada: w1*x1 + w2*x2 + ... (+ bias si corresponde)
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        
        // Aplicamos la función sigmoid
        return sigmoid(sum);
    }
    
    /**
     * Realiza la predicción (clasificación binaria)
     * @param inputs Vector de entradas
     * @return 1 si la salida es >= 0.5, 0 en caso contrario
     */
    public int predict(double[] inputs) {
        return calculateOutput(inputs) >= 0.5 ? 1 : 0;
    }
    
    /**
     * Entrena el perceptrón con un conjunto de datos
     * @param trainingData Matriz de datos de entrenamiento
     * @param targets Vector de salidas esperadas
     */
    public void train(double[][] trainingData, double[] targets) {
        int maxEpochs = 10000;
        double errorThreshold = 0.01;
        
        int epoch = 0;
        boolean converged = false;
        
        System.out.println("Iniciando entrenamiento del perceptron " + 
                          (useBias ? "con sesgo..." : "sin sesgo..."));
        System.out.println("Pesos iniciales: " + weightsToString());
        if (useBias) {
            System.out.println("Sesgo inicial: " + String.format("%.4f", bias));
        }
        
        // Entrenamiento por épocas
        while (!converged && epoch < maxEpochs) {
            double totalError = 0;
            
            // Procesamos cada ejemplo de entrenamiento
            for (int i = 0; i < trainingData.length; i++) {
                double[] inputs = trainingData[i];
                double target = targets[i];
                
                // Calculamos la salida actual
                double output = calculateOutput(inputs);
                
                // Calculamos el error
                double error = target - output;
                totalError += Math.pow(error, 2); // Error cuadrático
                
                // Factor derivada de la función sigmoid: output * (1 - output)
                double sigmoidDerivative = output * (1 - output);
                
                // Actualizamos los pesos
                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * sigmoidDerivative * inputs[j];
                }
                
                // Actualizamos el sesgo si corresponde
                if (useBias) {
                    bias += learningRate * error * sigmoidDerivative;
                }
            }
            
            // Calculamos el error promedio
            double mse = totalError / trainingData.length;
            epoch++;
            
            // Verificamos si hemos convergido
            if (mse < errorThreshold) {
                converged = true;
            }
            
            // Mostramos progreso cada 100 épocas
            if (epoch % 100 == 0) {
                System.out.println("Epoca " + epoch + ", Error: " + mse);
            }
        }
        
        System.out.println("Entrenamiento completado en " + epoch + " epocas.");
        System.out.println("Pesos finales: " + weightsToString());
        if (useBias) {
            System.out.println("Sesgo final: " + String.format("%.4f", bias));
        }
        
        // Mostramos resultados detallados
        printResults(trainingData, targets);
    }
    
    /**
     * Convierte los pesos a una cadena para mostrarlos
     * @return String con los pesos formateados
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
    
    /**
     * Muestra los resultados detallados del perceptrón
     * @param data Datos de entrada
     * @param expectedOutputs Salidas esperadas
     */
    public void printResults(double[][] data, double[] expectedOutputs) {
        System.out.println("\n--- Resultados detallados ---");
        
        for (int i = 0; i < data.length; i++) {
            double[] inputs = data[i];
            double expected = expectedOutputs[i];
            
            // Calculamos la suma ponderada
            double weightedSum = useBias ? bias : 0;
            for (int j = 0; j < weights.length; j++) {
                weightedSum += inputs[j] * weights[j];
            }
            
            // Calculamos la salida usando la función sigmoid
            double output = sigmoid(weightedSum);
            int prediction = output >= 0.5 ? 1 : 0;
            
            // Mostramos los cálculos paso a paso
            System.out.print("Entradas: [");
            for (int j = 0; j < inputs.length; j++) {
                System.out.print((int)inputs[j]); // Convertimos a entero para simplificar
                if (j < inputs.length - 1) System.out.print(", ");
            }
            System.out.println("]");
            
            if (useBias) {
                System.out.println("Suma Ponderada = " + String.format("%.4f", weightedSum) + 
                                  " (incluye sesgo: " + String.format("%.4f", bias) + ")");
            } else {
                System.out.println("Suma Ponderada = " + String.format("%.4f", weightedSum));
            }
            
            System.out.println("Funcion de Activacion (Sigmoid) = " + String.format("%.4f", output));
            System.out.println("Prediccion: " + prediction + ", Esperado: " + (int)expected);
            System.out.println("------------------------");
        }
    }
}