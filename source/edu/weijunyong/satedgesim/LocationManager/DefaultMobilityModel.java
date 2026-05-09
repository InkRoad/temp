package edu.weijunyong.satedgesim.LocationManager;

import edu.weijunyong.satedgesim.DataCentersManager.ServersManager;
import edu.weijunyong.satedgesim.ScenarioManager.simulationParameters;

public class DefaultMobilityModel extends Mobility {


	public DefaultMobilityModel(Location currentLocation) {
		super(currentLocation);
	}

	public DefaultMobilityModel() { 
		super();
	}

	public Location getNextLocation(int ID, double Simulationtime, String type) {
		int time = scaledTimeIndex(Simulationtime, type); //double
		//String FID = Integer.toString(ID);
		//String fileName = null;
		double[] locationPos;
		if (type == "cloud") {
			//fileName = MainApplication.getLocationFolder() + "cloud/cloud Fixed Position.csv";
			locationPos= ServersManager.Setnodelocation(simulationParameters.Cloudlocationinfo,ID,time);
		} else if (type == "edge") {
			//fileName = MainApplication.getLocationFolder() + "edge_datacenter/edge Fixed Position.csv";
			locationPos= ServersManager.Setnodelocation(simulationParameters.EdgeDataCenterslocationinfo,ID,time);
		} else {
			//fileName = MainApplication.getLocationFolder() + "edge_devices/mist Fixed Position.csv";
			locationPos= ServersManager.Setnodelocation(simulationParameters.EdgeDeviceslocationinfo,ID,time);
		}
		
    	
    	Double x_position = locationPos[0];
    	Double y_position = locationPos[1];
    	Double z_position = locationPos[2];
    	currentLocation = new Location(x_position, y_position, z_position);
    	//System.out.println("DefaultMobilityModel: "+type + FID+ " Location is: "+ x_position+","+y_position+","+z_position);
		return new Location(x_position, y_position, z_position);
	}

	private int scaledTimeIndex(double simulationTime, String type) {
		double speedScale = 1.0;
		if ("mist".equals(type) && simulationParameters.VEHICLE_SPEED > 0) {
			speedScale = simulationParameters.VEHICLE_SPEED / 60.0;
		}
		return Math.max(0, (int) Math.round(simulationTime * speedScale));
	}

	public Location getCurrentLocation() {
		return this.currentLocation;
	}
}
