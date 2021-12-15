if (waitInWarehouse.size() == 0 ) { traceln(red, "warning: no unit loads in warehouse");}

ArrayList<Pod> availablePods = new ArrayList<Pod>(); // initialize array lists for all unit loads currently in warehouse
ArrayList<Pallet> availablePallets = new ArrayList<Pallet>();

for (Unit unit: waitInWarehouse){ // go through each unit load and add it to separate array lists for pods and pallets
	if (unit.unitType == Pod){
		Pod p = (Pod)unit;
		if(p.partNumbers.contains(retrievalOrder.partNumber)){
			availablePods.add(p);
		}

	}
	else if (unit.unitType == Pallet){
		Pallet p = (Pallet)unit;
		if(p.partNumber == retrievalOrder.partNumber){
			availablePallets.add(p);
		}

	}
}
List<Pod> availablePodsFiltered = filter(availablePods, pod -> pod.currentStocks.get(pod.getIndex(retrievalOrder.partNumber)) != 0 ); 
// if the pod contains the part number but has 0 stock, then it is filtered out from available pods to choose from
List<Pod> sortedPods = sortAscending(availablePodsFiltered, pod -> pod.currentStocks.get(pod.getIndex(retrievalOrder.partNumber))); 
// sort pods by ascending order of currentStocks, this is so the function can choose partially full pods before full ones
List<Pallet> sortedPallets = sortAscending(availablePallets, pallet -> pallet.currentStock);
// same as for pods
if (showStandardTraceln == true) {
	traceln("sorted Pods: " + sortedPods);
	traceln("sorted Pallets: " + sortedPallets);
}
 
//int i = 0;
int tempNumberParts = retrievalOrder.numberParts; // temp variable to keep track of how many total parts a retrieval order needs
while (tempNumberParts > 0 ){ // while retrieval order still needs parts
	UnitType unitType = getUnitType(retrievalOrder.partNumber, "partNumber");
	if (unitType == Pod){
		Pod selectedPod = new Pod(); // initialize selectedPod before it's used in the if and else if blocks
		if (retrievalStrategy == LowestStockFirst){ // if following LowestStockFirst retrievalStrategy, get the pod with the lowest stock and remove it after so it won't be selected again
			selectedPod = sortedPods.get(0);
			sortedPods.remove(0);
			}
		else if (retrievalStrategy == RandomRetrieval){ // if following Random retrievalStrategy, get a random pod and remove it after
			int randomInt = uniform_discr(0, (sortedPods.size()-1));
			selectedPod = sortedPods.get(randomInt);
			sortedPods.remove(selectedPod);
		}
			int podCurrentStock = selectedPod.currentStocks.get(selectedPod.getIndex(retrievalOrder.partNumber));
			if (tempNumberParts > podCurrentStock) {
				selectedPod.partsToBePicked = podCurrentStock;
				}
			else {
				selectedPod.partsToBePicked = tempNumberParts;
				}
			tempNumberParts = tempNumberParts - podCurrentStock; // subtract the selected pod's stock from the tempNumberParts
			
			selectedPod.currentRetrievalOrder = retrievalOrder; // assign the retrievalOrder to the selectedPod's variable
			waitInWarehouse.free(selectedPod); // let selectedPod out of the waitInWarehouse wait block
			if (showStandardTraceln == true) traceln("found: " + selectedPod);
		
		
	}
	else if (unitType == Pallet){ // pretty much same but for pallets
		Pallet selectedPallet = new Pallet();
		if (retrievalStrategy == LowestStockFirst){
			selectedPallet = sortedPallets.get(0); // get first element (will be lowest stock)
			if (showStandardTraceln == true) traceln("selecting " + selectedPallet.color);
			sortedPallets.remove(0); // and remove first element
		}
		else if (retrievalStrategy == RandomRetrieval){
			int randomInt = uniform_discr(0, (sortedPallets.size()-1));
			selectedPallet = sortedPallets.get(randomInt); //get a random element
			sortedPallets.remove(selectedPallet); // and remove it
		}
			int palletCurrentStock = selectedPallet.currentStock;
			if (tempNumberParts > palletCurrentStock) {
				selectedPallet.partsToBePicked = palletCurrentStock;
				}
			else {
				selectedPallet.partsToBePicked = tempNumberParts;
				}
			tempNumberParts = tempNumberParts - palletCurrentStock; // update remaining parts needed for retrieval order
			selectedPallet.currentRetrievalOrder = retrievalOrder; // add retrievalOrder to selectedPallet
			
			int palletsOnTile = selectedPallet.currentTile.currentUnits.size();
			int palletsAboveSelected = palletsOnTile - selectedPallet.stackLevel - 1; // calculates how many pallets are on top of the selectedPallet 
			if (showStandardTraceln == true) traceln("palletsAboveSelected " + palletsAboveSelected);
			for (int i = palletsOnTile; i > palletsOnTile - palletsAboveSelected; i--){
				// loop through the pallets that are on top and free them from waitInWarehouse
				// starting with the pallets on top, so that they can be relocated
				Pallet palletToBeRelocated = (Pallet)selectedPallet.currentTile.currentUnits.get(i-1);
				palletToBeRelocated.toBeRelocated = true;
				waitInWarehouse.free(palletToBeRelocated); 
			  	if (showStandardTraceln == true) traceln("relocating " + palletToBeRelocated.color + " ID" + palletToBeRelocated.palletID);
			}
			
			waitInWarehouse.free(selectedPallet); 
			if (showStandardTraceln == true) traceln("found: " + selectedPallet);
		
	}
}
