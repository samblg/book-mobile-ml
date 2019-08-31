#include "CrawlerTopology.h"
#include "SpiderTaskSpout.h"
#include "UrlParseBolt.h"
#include "HashFilterBolt.h"

#include "hurricane/topology/Topology.h"

hurricane::topology::Topology* GetTopology() {
    hurricane::topology::Topology* topology = new hurricane::topology::Topology("crawler-topology");

    topology->SetSpout("get-task-spout", new SpiderTaskSpout)
        .ParallismHint(1);

    topology->SetBolt("url-parse-bolt", new UrlParseBolt)
        .Random("get-task-spout")
        .ParallismHint(3);

    topology->SetBolt("hash-filter-bolt", new HashFilterBolt)
        .Random("url-parse-bolt")
        .ParallismHint(2);

    return topology;
}
