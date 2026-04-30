class Geofence:
    def __init__(self, polygon_points=None):
        """
        Initializes the Geofence.
        :param polygon_points: List of (x, y) tuples defining the geofence polygon.
        """
        self.polygon_points = polygon_points or []

    def set_polygon(self, points):
        self.polygon_points = points

    def is_point_inside(self, x, y):
        """
        Ray-casting algorithm to determine if a point is inside the polygon.
        """
        if len(self.polygon_points) < 3:
            return False # Not a valid polygon

        n = len(self.polygon_points)
        inside = False

        p1x, p1y = self.polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def check_breach(self, object_centroid_x, object_centroid_y):
        """
        Checks if the object's centroid has breached the geofence.
        """
        if not self.polygon_points:
            return False
            
        # A breach occurs if the object LEAVES the geofenced area
        return not self.is_point_inside(object_centroid_x, object_centroid_y)
