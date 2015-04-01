package network;

import java.util.ArrayList;
import java.util.LinkedList;

public class NeuralNetwork
{
	/** Rozmiar warstwy wejsciowej sieci */ 
	private int inputLayerSize;
	
	/** Rozmiar warstwy ukrytej sieci */
	private int	hiddenLayerSize;
	
	/** Rozmiar warstwy wyjsciowej sieci */
	private int outputLayerSize;
	
	public LinkedList<ArrayList<Neuron>> network;
	
	private void addNeuronToNetword(int layerSize, int numberOfNeuronInputs)
	{
		ArrayList<Neuron> tmpArray = new ArrayList<Neuron>();
		for (int i = 0; i < layerSize; i++)
		{
			Neuron tmpNeuron = new Neuron(numberOfNeuronInputs);
			tmpArray.add(tmpNeuron);
		}
		this.network.add(tmpArray); 
	}
	
	/** Konstruktor trój-argumentowy klasy NeuralNetwork 
	@param int inputLayer Rozmiar warstwy wejsciowej sieci
	@param int hiddenLayer Rozmiar warstwy ukrytej sieci
	@param int outputLayer Rozmiar warstwy wyjsciowej sieci
	*/
	public NeuralNetwork(int inputLayer, int hiddenLayer, int outputLayer)
	{
		this.inputLayerSize = inputLayer;
		this.hiddenLayerSize = hiddenLayer;
		this.outputLayerSize = outputLayer;
		this.network = new LinkedList<ArrayList<Neuron>>();
	}
	
	/** Metoda odpowiedzialna, za stworzenie siecii neuronowej */
	public void CreateNetwork()
	{
		addNeuronToNetword(this.inputLayerSize, 1);
		addNeuronToNetword(this.hiddenLayerSize, 2);
		addNeuronToNetword(this.outputLayerSize, 5);
	}
}
