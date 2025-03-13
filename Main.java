/**
 * Clase principal para ejecutar los ejemplos de perceptrones
 */
public class Main {
    public static void main(String[] args) {
        // Datos para entrenar la compuerta lógica AND
        double[][] andTrainingData = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        // Salidas esperadas para la compuerta AND
        double[] andTargets = {0, 0, 0, 1};
        
        // Datos para entrenar la compuerta lógica OR
        double[][] orTrainingData = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        // Salidas esperadas para la compuerta OR
        double[] orTargets = {0, 1, 1, 1};
        
        // Crear y entrenar un perceptrón SIN sesgo para la compuerta AND
        System.out.println("=== PERCEPTRON SIN SESGO PARA COMPUERTA AND ===");
        // Parámetros: 2 entradas, tasa de aprendizaje 0.1, sin sesgo (false)
        Perceptron perceptronAnd = new Perceptron(2, 0.1, false);
        perceptronAnd.train(andTrainingData, andTargets);
        
        // Crear y entrenar un perceptrón CON sesgo para la compuerta OR
        System.out.println("\n\n=== PERCEPTRON CON SESGO PARA COMPUERTA OR ===");
        // Parámetros: 2 entradas, tasa de aprendizaje 0.1, con sesgo (true)
        Perceptron perceptronOr = new Perceptron(2, 0.1, true);
        perceptronOr.train(orTrainingData, orTargets);
    }
}