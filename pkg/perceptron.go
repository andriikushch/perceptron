package perceptron

import (
	"errors"
	"fmt"
	"golang.org/x/exp/constraints"
	"math/rand/v2"
)

// NewPerceptron creates a new Perceptron instance with given features and labels.
// It initializes weights to 1 and bias to 0.
func NewPerceptron[T constraints.Float](features [][]T, labels []T) (*Perceptron[T], error) {
	// Check if there is at least one feature
	if len(features) < 1 {
		return nil, errors.New("cannot create perceptron: no features provided")
	}

	// Check if the number of labels matches the number of feature samples
	if len(features) != len(labels) {
		return nil, errors.New("cannot create perceptron: number of labels does not match number of feature samples")
	}

	// Initialize weights to 1 for each feature dimension
	var weights []T
	for range features[0] {
		weights = append(weights, T(1))
	}

	// Initialize bias to 0
	bias := T(0)

	// Return the initialized Perceptron
	return &Perceptron[T]{
		weights:  weights,
		bias:     bias,
		features: features,
		labels:   labels,
	}, nil
}

// Perceptron represents a simple linear classifier for numeric data
type Perceptron[T constraints.Float] struct {
	// weights are the adjustable coefficients for each input feature
	weights []T

	// bias is an extra adjustable value to shift the decision boundary
	bias T

	// features stores the training data (each inner slice is one data sample)
	features [][]T

	// labels stores the expected outputs (true classes) for the training data
	labels []T

	// mErr (mean error) tracks the average prediction error during training
	mErr T

	// trainingHistory keeps a record of weights, bias and error after each training step (optional, useful for analysis or visualization)
	trainingHistory [][]T
}

// Train trains the Perceptron using the given learning rate and number of epochs.
// If withHistory is true, it saves weights, bias, and mean error after each update for later analysis.
func (p *Perceptron[T]) Train(learningRate T, epoch int, withHistory bool) ([]T, T, T, error) {
	var err error

	// Repeat the training process for the specified number of epochs
	for range epoch {
		// Calculate the mean error over the current dataset
		p.mErr, err = p.meanError()
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to calculate mean error: %w", err)
		}

		// Pick a random training sample (feature and label)
		i := rand.IntN(len(p.features))
		randomFeature := p.features[i]
		featureLabel := p.labels[i]

		// Update the weights and bias based on the selected sample
		err = p.updateWeights(randomFeature, featureLabel, learningRate)
		if err != nil {
			return nil, 0, 0, fmt.Errorf("failed to update weights: %w", err)
		}

		// Optionally save the training history (weights, bias, error) after each update
		if withHistory {
			historyRecord := make([]T, len(p.weights)+2)
			copy(historyRecord, p.weights)
			historyRecord[len(p.weights)] = p.bias
			historyRecord[len(p.weights)+1] = p.mErr

			p.trainingHistory = append(p.trainingHistory, historyRecord)
		}
	}

	// Return final weights, bias, and mean error after training
	return p.weights, p.bias, p.mErr, nil
}

// Predict returns class of the feature or error if any
func (p *Perceptron[T]) Predict(feature []T) (T, error) {
	prediction, _, err := p.predict(feature)

	return prediction, err
}

// TrainingHistory returns trainingHistory
func (p *Perceptron[T]) TrainingHistory() [][]T {
	return p.trainingHistory
}

// score calculates the raw score (weighted sum + bias) for a given feature vector.
// It computes the dot product of weights and features, then adds the bias.
// Returns: the score and any potential error during dot product calculation.
func (p *Perceptron[T]) score(features []T) (T, error) {
	// Compute the dot product between weights and features
	res, err := p.dot(p.weights, features)
	if err != nil {
		return 0, fmt.Errorf("failed to calculate dot product in score: %w", err)
	}

	// Add the bias to the dot product result
	return res + p.bias, nil
}

// predict calculates the prediction for a given feature vector.
// It first computes the raw score (linear combination of inputs and weights + bias).
// Then, it applies the step function to determine the final prediction (e.g., 0 or 1).
// Returns: prediction, score, and possible error.
func (p *Perceptron[T]) predict(feature []T) (T, T, error) {
	// Calculate the raw score (dot product + bias)
	s, err := p.score(feature)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to calculate score in predict: %w", err)
	}

	// Apply step function to turn score into a discrete prediction
	prediction := p.stepFunc(s)

	return prediction, s, nil
}

// errorFunc calculates the error for a single feature-label pair.
// It returns 0 if the prediction is correct; otherwise, it returns the absolute value of the score.
// This error is used for updating weights during training.
func (p *Perceptron[T]) errorFunc(feature []T, label T) (T, error) {
	// Predict the label and get the raw score for the given feature
	prediction, score, err := p.predict(feature)
	if err != nil {
		return 0, fmt.Errorf("error during prediction in errorFunc: %w", err)
	}

	// If prediction matches the true label, there is no error
	if prediction == label {
		return 0, nil
	}

	// If prediction is incorrect, return the absolute score as the error
	return p.abs(score), nil
}

// meanError calculates the mean (average) error over all training samples.
// It uses the Perceptron's error function (errorFunc) for each feature-label pair.
func (p *Perceptron[T]) meanError() (T, error) {
	var res T

	// Loop over all training samples
	for i := range p.features {
		// Calculate the error for a single sample
		e, err := p.errorFunc(p.features[i], p.labels[i])
		if err != nil {
			return 0, fmt.Errorf("error evaluating error function at sample %d: %w", i, err)
		}
		// Accumulate the error
		res += e
	}

	// Return the mean error (total error divided by number of samples)
	return res / T(len(p.features)), nil
}

// updateWeights updates the perceptron's weights and bias based on the prediction error.
//
// It uses the perceptron learning rule:
//
//	weight = weight + learningRate * (label - prediction) * feature
//	bias   = bias + learningRate * (label - prediction)
//
// Parameters:
// - feature: the input feature vector
// - label: the true label (expected output)
// - learningRate: a small value that controls how much the weights are adjusted
//
// Returns an error if prediction fails.
func (p *Perceptron[T]) updateWeights(feature []T, label T, learningRate T) error {
	// Predict the output for the given input features
	prediction, err := p.Predict(feature)
	if err != nil {
		return fmt.Errorf("failed to predict: %w", err)
	}

	// Update each weight based on the difference between label and prediction
	for i := range p.weights {
		p.weights[i] += (label - prediction) * feature[i] * learningRate
	}

	// Update the bias term similarly
	p.bias += (label - prediction) * learningRate

	return nil
}

// stepFunc applies a step activation function to the given value.
// It returns 1 if the input value is greater than 0, and 0 otherwise.
// This is used to make a binary decision in the perceptron: class 1 or class 0.
func (p *Perceptron[T]) stepFunc(v T) T {
	// If the value is greater than 0, return 1 (indicating class 1)
	if v > 0 {
		return 1
	}

	// If the value is less than or equal to 0, return 0 (indicating class 0)
	return 0
}

// abs returns the absolute value of the given number n.
// If n is positive or zero, it returns n directly.
// If n is negative, it returns -n.
func (p *Perceptron[T]) abs(n T) T {
	if n >= 0 {
		return n
	}
	return -n
}

// dot calculates the dot product of two vectors v1 and v2.
// The dot product is the sum of element-wise multiplications: (v1[0] * v2[0]) + (v1[1] * v2[1]) + ...
// Returns: the scalar result and any potential error if vector sizes do not match.
func (p *Perceptron[T]) dot(v1, v2 []T) (T, error) {
	// Ensure the vectors have the same length
	if len(v1) != len(v2) {
		return 0, fmt.Errorf("cannot calculate dot product: vectors must be of the same length (got %d and %d)", len(v1), len(v2))
	}

	var result T
	// Sum up the element-wise products
	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}

	return result, nil
}
