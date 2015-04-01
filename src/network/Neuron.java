package network;

import java.util.ArrayList;

public class Neuron
{
	/** Dolne ograniczenie przedzia³u wag */
	private static final double LOWER_LIMIT_WEIGHTS_RANGE = -0.5;
	
	/** Gorne ograniczenie przedzia³u wag */
	private static final double UPPER_LIMIT_WEIGHTS_RANGE = 0.5;
	
	/** Iloœæ wejœæ w neuronie */
	private int numberOfNeuronInputs;
	
	/** Wylosowana waga wejœcia dla neuronu */
	private double weight;
	
	/** Wartoœæ bias (dobierana na pocz¹tku losowo) */
	private double bias;
	
	/** Wartoœæ funkcji wyjœcia z neuronu czyli suma ka¿dego z wejœæ razy odpowiednia waga przypisana temu wejœciu */
	private double outputFunction;
	
	/** Tablica zawierajaca wagê ka¿dego wejœcia neuronu */
	private ArrayList<Double> weightsOfNeuronInputs;
	
	/** Gradient klasy neuron */ 
	private double gradient;

	/** Konstruktor klasy Neuron 
	@param int numberOfNeuronInputs Iloœæ wejœæ pojedynczego neuronu */
	public Neuron(int numberOfNeuronInputs)
	{
		this.weightsOfNeuronInputs = new ArrayList<Double>();
		this.numberOfNeuronInputs = numberOfNeuronInputs;
		this.outputFunction = 0.0;
		this.weight = 0.0;
		this.bias = 0.0;
	}
	
	public double getWeight()
	{
		return weight;
	}

	public void setWeight(double weight)
	{
		this.weight = weight;
	}

	public double getBias()
	{
		return bias;
	}

	public void setBias(double bias)
	{
		this.bias = bias;
	}

	public double getOutputFunction()
	{
		return outputFunction;
	}

	public void setOutputFunction(double outputFunction)
	{
		this.outputFunction = outputFunction;
	}

	public ArrayList<Double> getWeightsOfNeuronInputs()
	{
		return weightsOfNeuronInputs;
	}

	public void setWeightsOfNeuronInputs(ArrayList<Double> weightsOfNeuronInputs)
	{
		this.weightsOfNeuronInputs = weightsOfNeuronInputs;
	}
	
	public double getOneWeight(int i)
	{
		return this.weightsOfNeuronInputs.get(i);
	}
	
	public void setOneWeight(int i, double weight)
	{
		this.weightsOfNeuronInputs.set(i, weight);
	}

	public double getGradient()
	{
		return gradient;
	}

	public void setGradient(double gradient)
	{
		this.gradient = gradient;
	}
		
	/** 
	 * Metoda oblicza wagi dla ka¿dego z wejœæ pojedynczego neuronu 
	 * @return true gdy uda siê obliczyæ wagi dla wszystkich wejœæ neuronu.
	 * W przeciwnym wypadku wartosci¹ zwracan¹ jest false
	*/
	public boolean calculateNeuronInputsWeights()
	{
		if (this.numberOfNeuronInputs <= 0)
		{
			return false;
		}
		else
		{
			for (int i = 0; i < this.numberOfNeuronInputs; i++)
			{
				this.weight = (UPPER_LIMIT_WEIGHTS_RANGE - LOWER_LIMIT_WEIGHTS_RANGE) 
					* (Math.random()) + LOWER_LIMIT_WEIGHTS_RANGE;
				weightsOfNeuronInputs.add(this.weight);
			}
			return true;
		}
	}

	/**
	 * Metoda oblicza wartoœæ funkcji wyjœcia z neuronu 
	 * @param std::vector<double> neuronInputData Tablica zawieraj¹ca dane wejœciowe
	 * podawane na kolejne wejœcia pojedynczego neuronu
	 */
	public void calculateNeuronOutputFunction(ArrayList<Double> neuronInputData)
	{
		for (int i = 0; i < numberOfNeuronInputs; i++)
		{
			this.outputFunction += ((neuronInputData.get(i) * this.weightsOfNeuronInputs.get(i)) + bias);
		}
		this.outputFunction = outputFunction / 1000;
	}
}
