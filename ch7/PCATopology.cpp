#include "PCATopology.h"
#include "DataReaderSpout.h"
#include "PCABolt.h"

#include "hurricane/topology/Topology.h"

hurricane::topology::Topology* GetTopology() {
    hurricane::topology::Topology* topology = new hurricane::topology::Topology("pca-topology");

    topology->SetSpout("data-reader-spout", new DataReaderSpout)
        .ParallismHint(1);

    topology->SetBolt("pca-bolt", new PCABolt)
        .Random("data-reader-spout")
        .ParallismHint(3);

    return topology;
}
