import java.util.ArrayList;
import network.BackPropagation;
import network.NeuralNetwork;

public class NetworkManager
{
	/** Liczba neuron�w w warstwie wej�ciowej sieci */
	private static final int INPUT_LAYER_NEURONS_NUMBER = 2;
	
	/** Liczba neuron�w w warstwie ukrytej sieci */
	private static final int HIDDEN_LAYER_NEURONS_NUMBER = 5;
	
	/** Liczba neuron�w w warstwie wyj�ciowej sieci */
	private static final int OUTPUT_LAYER_NEURONS_NUMBER = 1;

	/** Tablica z danymi wczytanymi z pliku */
	private ArrayList<Double> inputData;

	/** Tablica przechowuj�ca sugerowane wyniki */
	private ArrayList<Double> predictedNetworkAnswer;

	/** Zmienna wska�nikowa wskazuj�ca obiekt NeuralNetwork */
	private NeuralNetwork network;

	/** Wska�nik na obiekt, typu BackPropagation */
	private BackPropagation propagation;

	/** Zmienne tymczasowe na warto�ci z pliku */
	private double temp1, temp2, temp3;
	
	/** Zmienne pomocnicze */ 
	private double test, p1;

	/** Metoda s�u��ca do stworzenia sieci neuronowej */
	private void prepareNetwork()
	{
		this.network.CreateNetwork();
	}

	/** Metoda ucz�ca sie� neuronow� */
	private void teachNetwork()
	{
		int i = 0;
		while ((i+2) < this.inputData.size())
		{
			temp1 = this.inputData.get(i);
			temp2 = this.inputData.get(i+1);
			temp3 = this.inputData.get(i+2);
			p1 = temp3;
			propagation.BackPropagationMethod(temp1, temp2, temp3, network);
			i++;
			test = propagation.getEndResult();
		}
		// przygotowanie danych do metody Calculate (by tam ponownie nie iterowa� po tablicy)
		temp1 = this.inputData.get(i);
		temp2 = this.inputData.get(i+1);
	}

	/** Metoda obliczaj�ca odpowied� sieci na podstawie danych wej�ciowych
	@retun double Odpowied� sieci 
	*/
	private double predictNetworkAnswer()
	{
		// temp1 i temp2 zosta�y uprzednio przygotowane przez metod� TeachNetwork, po zako�czeniu jej p�tli g��wnej 
		propagation.BackPropagationMethod(temp1, temp2, 0, network);
		return propagation.getEndResult();
	}

	/** Konstruktor klasy Networkmanager */
	public NetworkManager()
	{
		this.inputData = new ArrayList<Double>();
		this.predictedNetworkAnswer = new ArrayList<Double>();
		this.network = new NeuralNetwork(NetworkManager.INPUT_LAYER_NEURONS_NUMBER, 
				NetworkManager.HIDDEN_LAYER_NEURONS_NUMBER, 
				NetworkManager.OUTPUT_LAYER_NEURONS_NUMBER);
		this.propagation = new BackPropagation(NetworkManager.INPUT_LAYER_NEURONS_NUMBER, 
				NetworkManager.HIDDEN_LAYER_NEURONS_NUMBER, 
				NetworkManager.OUTPUT_LAYER_NEURONS_NUMBER); 
	}
	
	public ArrayList<Double> getPredictedNetworkAnswer()
	{
		return this.predictedNetworkAnswer;
	}

	public void setInputData(ArrayList<Double> data)
	{
		this.inputData.addAll(data);
	}

	/** Metoda uruchamiaj�ca proces tworzenia i uczenia sieci */
	public void runNetwork()
	{
		this.prepareNetwork();
		this.teachNetwork();
		double finalyNetworkResult = predictNetworkAnswer();
		finalyNetworkResult = (p1 * finalyNetworkResult) / test;
		this.predictedNetworkAnswer.add(finalyNetworkResult);
	}
}
