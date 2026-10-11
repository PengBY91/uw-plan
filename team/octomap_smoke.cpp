#include <octomap/OcTree.h>
#include <iostream>

int main() {
    octomap::OcTree map(0.1);
    const octomap::point3d occupied(1.0, 0.0, 0.0);
    const octomap::point3d free(0.0, 1.0, 0.0);
    const octomap::point3d unknown(2.0, 2.0, 2.0);
    map.updateNode(occupied, true);
    map.updateNode(free, false);
    const auto* hit = map.search(occupied);
    const auto* miss = map.search(free);
    if (!hit || !map.isNodeOccupied(hit)) return 1;
    if (!miss || map.isNodeOccupied(miss)) return 2;
    if (map.search(unknown) != nullptr) return 3;
    std::cout << "occupied/free/unknown: PASS; resolution=" << map.getResolution() << " m\n";
    return 0;
}
