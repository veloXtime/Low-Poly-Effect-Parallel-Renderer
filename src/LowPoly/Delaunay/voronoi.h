#ifndef VORONOI_H
#define VORONOI_H

struct Point {
    double x, y;
    
    Point(double x, double y) : x(x), y(y) {}
};

struct Edge {
    Point *start, *end;  // endpoints of the edge
    Point *direction;    // Direction vector for infinite edges
    Point *left, *right; // The sites defining the bisector

    Edge(Point* start, Point* end) : start(start), end(end) {}
};

struct Cell {
    Point *site;                 // The site corresponding to this cell
    std::vector<Edge*> edges;    // Edges bounding the cell

    Cell(Point *site) : site(site) {}
};

struct Event {
    int type; // 0 for site event, 1 for circle event
    Point p; 
}

class VoronoiDiagram {
private:
    std::vector<Point> sites;
    std::priority_queue<Event> events;
    Node 

public:
    void addSiteEvent(Point site);

    void processEvents() {
        while (!events.empty()) {
            Event e = events.top();
            events.pop();
            e.handle(); // Execute the event's associated handler
        }
    }

    void setupEventHandlers() {
        // Define handlers for site events and circle events here
    }
};


#endif