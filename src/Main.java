import java.util.ArrayList;

public class Main
{
	/**
	 * @param args
	 */
	public static void main(String[] args)
	{	
		ArrayList<Double> data = new ArrayList<Double>();
		for (int i = 10; i < 100; i+= 10)
		{
			data.add((double) i);
		}
		NetworkManager networkManager = new NetworkManager();
		networkManager.setInputData(data);
		networkManager.runNetwork();
		System.out.println(networkManager.getPredictedNetworkAnswer().get(0));
	}

}
