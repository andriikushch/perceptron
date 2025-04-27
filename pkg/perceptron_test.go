package perceptron

import (
	"fmt"
	"log"
)

func ExamplePerceptron_Train_two_dimensional_case() {
	// Define the training features (input vectors)
	features := [][]float64{
		// Class 0 (label 0)
		{1, 0}, {0, 2}, {1, 1}, {1, 2}, {0, 0},
		{0, 1}, {2, 0}, {2, 1}, {3, 0}, {0, 3},
		{1.5, 1.5}, {0.5, 1}, {1, 0.5}, {0.2, 1.8}, {1.2, 1.3},
		{2.5, 0.5}, {0.3, 2.3}, {1.1, 0.9}, {1.8, 0.2}, {0.7, 1.1},

		// Class 1 (label 1)
		{1, 3}, {2, 2}, {2, 3}, {3, 2}, {2, 4},
		{3, 3}, {4, 2}, {4, 3}, {3, 4}, {5, 3},
		{3.5, 2.5}, {2.5, 3.5}, {4.2, 2.8}, {3.1, 3.9}, {2.2, 4.1},
		{3.3, 2.7}, {4.4, 3.6}, {2.6, 3.2}, {3.9, 2.1}, {3.7, 3.3},
	}

	// Define the corresponding labels for each feature vector
	labels := []float64{
		// Class 0
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		// Class 1
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
	}

	// Initialize a new perceptron with the training data
	p, err := NewPerceptron[float64](features, labels)
	if err != nil {
		log.Fatalf("failed to create perceptron: %v", err)
	}

	// Train the perceptron
	// Parameters:
	// - learning rate: 0.01
	// - epochs: 250
	// - verbose: true (prints progress during training)
	weights, bias, meanError, err := p.Train(0.01, 250, true)
	if err != nil {
		log.Fatalf("failed to train perceptron: %v", err)
	}

	if bias < -1 && bias > 1 {
		log.Println("bias seems to have a wrong value")
	}

	if (weights[0] < -0.3 && weights[0] > 0.3) || (weights[1] < -0.3 && weights[1] > 0.3) {
		log.Println("weights seems to have a wrong value")
	}

	fmt.Printf("Mean error: %.6f\n", meanError)
	// Output: Mean error: 0.000000
}

func ExamplePerceptron_Train_three_dimensional_case() {
	// Define the training features (input vectors) in 3D space
	features := [][]float64{
		// Class 0 (label 0)
		{1, 0, 0}, {0, 2, 0}, {1, 1, 0}, {1, 2, 0}, {0, 0, 0},
		{0, 1, 1}, {2, 0, 0}, {2, 1, 1}, {3, 0, 1}, {0, 3, 0},
		{1.5, 1.5, 0.5}, {0.5, 1, 0.5}, {1, 0.5, 0.2}, {0.2, 1.8, 0.3}, {1.2, 1.3, 0.4},
		{2.5, 0.5, 0.2}, {0.3, 2.3, 0.1}, {1.1, 0.9, 0.3}, {1.8, 0.2, 0.2}, {0.7, 1.1, 0.4},

		// Class 1 {label 1}
		{1, 3, 2}, {2, 2, 2}, {2, 3, 2}, {3, 2, 2}, {2, 4, 3},
		{3, 3, 2}, {4, 2, 3}, {4, 3, 3}, {3, 4, 3}, {5, 3, 3},
		{3.5, 2.5, 2.5}, {2.5, 3.5, 2.5}, {4.2, 2.8, 2.7}, {3.1, 3.9, 3.0}, {2.2, 4.1, 2.8},
		{3.3, 2.7, 2.6}, {4.4, 3.6, 3.2}, {2.6, 3.2, 2.4}, {3.9, 2.1, 2.3}, {3.7, 3.3, 2.9},
	}

	// Define the corresponding labels for each feature vector
	labels := []float64{
		// Class 0
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		// Class 1
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
	}

	// Initialize a new perceptron with the training data
	p, err := NewPerceptron[float64](features, labels)
	if err != nil {
		log.Fatalf("failed to create perceptron: %v", err)
	}

	// Train the perceptron
	// Parameters:
	// - learning rate: 0.01
	// - epochs: 300
	// - verbose: true (prints progress during training)
	weights, bias, meanError, err := p.Train(0.01, 300, true)
	if err != nil {
		log.Fatalf("failed to train perceptron: %v", err)
	}

	if bias < -1 && bias > 1 {
		log.Println("bias seems to have a wrong value")
	}

	if (weights[0] < -0.3 && weights[0] > 0.3) ||
		(weights[1] < -0.3 && weights[1] > 0.3) ||
		(weights[2] < -0.3 && weights[2] > 0.3) {
		log.Println("weights seems to have a wrong value")
	}

	fmt.Printf("Mean error: %.6f\n", meanError)
	// Output: Mean error: 0.000000
}
