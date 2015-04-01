package network;

import java.util.ArrayList;

public class BackPropagation
{
	/** Rozmiar warstwy wejsciowej sieci */
	private int inputLayerSize;

	/** Rozmiar warstwy ukrytej sieci */
	private int hiddenLayerSize;

	/** Rozmiar warstwy wyjsciowej sieci */
	private int outputLayerSize;

	/** Wspó³czynnik uczenia sieci */
	private double eta;

	/** Wynik koñcowy */
	private double endResult;

	/**
	 * Metoda oblicza gradient dla warstwy wyjœciowej sieci neuronowej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 * @param double destinationValue Wartoœæ docelowa
	 */
	private void computeGradientsForOutputLayer(NeuralNetwork net,
			double destinationValue)
	{
		// Compute output gradients
		ArrayList<Neuron> outputLayer = net.network.getLast();
		for (int i = 0; i < this.outputLayerSize; i++)
		{
			double networkAnswear = outputLayer.get(i).getOutputFunction();
			double derivative = (1 - networkAnswear) * networkAnswear;
			outputLayer.get(i).setGradient(
					derivative * (destinationValue - networkAnswear));
		}
	}

	/**
	 * Metoda oblicza gradient dla warstwy ukrytej sieci neuronowej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 */
	private void computeGradientsForHiddenLayer(NeuralNetwork net)
	{
		// Compute hidden gradients
		double sum = 0;
		double hiddenAnswear = 0.0;
		double derivative = 0.0;
		for (int i = 0; i < this.hiddenLayerSize; i++)
		{
			hiddenAnswear = net.network.get(1).get(i).getOutputFunction();
			ArrayList<Double> numbOfInputs = net.network.getLast().get(0).getWeightsOfNeuronInputs();
			derivative = (1 - hiddenAnswear) * hiddenAnswear;

			for (int j = 0; j < numbOfInputs.size(); j++)
			{
				sum += net.network.getLast().get(0).getGradient()
						* numbOfInputs.get(j);
			}
			net.network.get(1).get(i).setGradient(derivative * sum);
		}
	}

	/**
	 * Metoda wprowadza poprawki w wagach neuronów w warstwie ukrytej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 */
	private void updateNeuronsWeightsInHiddenLayer(NeuralNetwork net)
	{
		// Update hidden neuron weights
		// Iteracja po iloœci wejœc w neuronach warstwy ukrytej
		for (int i = 0; i < this.inputLayerSize; i++)
		{
			// Iteracja po warstwie ukrytej
			for (int j = 0; j < this.hiddenLayerSize; j++)
			{
				double delta = this.eta * net.network.get(1).get(j).getGradient()
						* net.network.getFirst().get(i).getOutputFunction();
				delta += net.network.get(1).get(j).getOneWeight(i);
				net.network.get(1).get(j).setOneWeight(i, delta);
			}
		}
	}

	/**
	 * Metoda wprowadza poprawki we wspó³czynnikach korekcyjnych w warstwie
	 * ukrytej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 */
	private void updateNeuronsBiasesInHiddenLayer(NeuralNetwork net)
	{
		// Update hidden neuron biases
		for (int i = 0; i < this.hiddenLayerSize; i++)
		{
			double delta = this.eta * net.network.get(1).get(i).getGradient();
			delta += net.network.get(1).get(i).getBias();
			net.network.get(1).get(i).setBias(delta);
		}
	}

	/**
	 * Metoda wprowadza poprawki w wagach neuronów w warstwie wyjœciowej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 */
	private void updateNeuronsWeightsInOutputLayer(NeuralNetwork net)
	{
		// Update output neurons weights
		for (int i = 0; i < this.hiddenLayerSize; i++)
		{
			for (int j = 0; j < this.outputLayerSize; j++)
			{
				double delta = eta * net.network.getLast().get(j).getGradient()
						* net.network.get(1).get(i).getOutputFunction();
				delta += net.network.getLast().get(j).getOneWeight(i);
				net.network.getLast().get(j).setOneWeight(i, delta);
			}
		}
	}

	/**
	 * Metoda wprowadza poprawki we wspó³czynnikach korekcyjnych w warstwie
	 * wyjœciowej
	 * 
	 * @param NeuralNetwork*
	 *            net WskaŸnik na sieæ neuronow¹, która wymaga nauki
	 */
	private void updateNeuronsBiasesInOutputLayer(NeuralNetwork net)
	{
		// 4b Update outputs neurons biases
		for (int i = 0; i < this.outputLayerSize; i++)
		{
			double delta = eta * net.network.getLast().get(i).getGradient();
			delta += net.network.getLast().get(i).getBias();
			net.network.getLast().get(i).setBias(delta);
		}
	}

	/** Konstruktor bezargumentowy klasy BackPropagation */
	public BackPropagation(int inputLayer, int hiddenLayer, int outputLayer)
	{
		eta = 0.3; // by³o 0.9
		this.inputLayerSize = inputLayer;
		this.hiddenLayerSize = hiddenLayer;
		this.outputLayerSize = outputLayer;
	}

	public double getEndResult()
	{
		return endResult;
	}

	public void BackPropagationMethod(double firstValue, double secondValue,
			double thirdValue, NeuralNetwork net)
	{
		double destinationValue = thirdValue;
		double afterConversion = 0;
		
		// tablica z wartosciami funkcji wyjscia neuronow z warstwy poprzedniej
		ArrayList<Double> neuronsResults = new ArrayList<>();
		
		// tablica do ustawienia wag neuronów wejœciowych na wartosc = 1
		ArrayList<Double> neutralizeWeights = new ArrayList<>(); 
		ArrayList<Double> parameters = new ArrayList<>();
		
		// konwersja destinationValues do przedzialu {0,1}
		destinationValue = destinationValue / (1000);

		for (int i = 0; i < this.inputLayerSize; i++)
		{
			// zapewnienie ze na 1 neuron poleci 1 wartosc z pliku a na 2 druga wartosc
			if (i == 0)
			{
				parameters.add(firstValue);
			} 
			else if (i == 1)
			{
				parameters.clear();
				parameters.add(secondValue);
			} 
			else
			{
				parameters.clear();
				parameters.set(0, 0.0);
			}

			neutralizeWeights.add(1.0); // ustawienie wag neuronów wejœciowych
										// na wart. = 1; jeden jako wartosc
										// neutralna
			ArrayList<Neuron> tmp = net.network.getFirst();//.get(i)
					//.setWeightsOfNeuronInputs(neutralizeWeights);
			tmp.get(i).setWeightsOfNeuronInputs(neutralizeWeights);
			net.network.getFirst().get(i)
					.calculateNeuronOutputFunction(parameters);
			afterConversion = net.network.getFirst().get(i).getOutputFunction();
			neuronsResults.add(afterConversion);
		}

		for (int i = 0; i < this.hiddenLayerSize; i++)
		{
			parameters.clear();
			parameters.add(neuronsResults.get(0));
			parameters.add(neuronsResults.get(1));
			net.network.get(1).get(i).calculateNeuronInputsWeights();
			net.network.get(1).get(i).calculateNeuronOutputFunction(parameters);
			afterConversion = net.network.get(1).get(i).getOutputFunction();
			neuronsResults.add(afterConversion);
		}

		for (int i = 0; i < this.outputLayerSize; i++)
		{
			parameters.clear();
			parameters.addAll(neuronsResults);
			net.network.getLast().get(i).calculateNeuronInputsWeights();
			net.network.getLast().get(i).calculateNeuronOutputFunction(parameters);
			afterConversion = net.network.getLast().get(i).getOutputFunction();
			neuronsResults.add(afterConversion);
		}

		if (thirdValue != 0)
		{
			// Back propagation
			this.computeGradientsForOutputLayer(net, destinationValue);
			this.computeGradientsForHiddenLayer(net);
			this.updateNeuronsWeightsInHiddenLayer(net);
			this.updateNeuronsBiasesInHiddenLayer(net);
			this.updateNeuronsWeightsInOutputLayer(net);
			this.updateNeuronsBiasesInOutputLayer(net);
		}
		this.endResult = net.network.getLast().get(0).getOutputFunction();
	}
}
