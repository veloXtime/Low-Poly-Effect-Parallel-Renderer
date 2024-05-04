#include <iostream>

#include "CImg.h"
#include "voronoi.h"

// The projector algorithm

void VoronoiDiagram::addEvent(Event &event) {
    events.emplace(event);

    // Create site event for every site
    for (auto & site: sites) {
        events.emplace(Event(0, site));
    }

    while (!this.events.empty()) {
        Event e = events.top();
        events.pop();
        if (e.type == 0) {
            handleSiteEvent(e);
        } else {
            handleCircleEvent(e);
        }
    }
}

void VoronoiDiagram::addParabola(Point & site) {
    
}


void 