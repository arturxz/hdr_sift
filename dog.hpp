struct KeyPoints {
	int x;			// coord x
	int y;			// coord y
	int z;			// col dogI
	int scale;		// lin dogI
	float response;	// intensidade do keypoint
};

static const int SIFT_MAX_INTERP_STEPS = 5;
static const int SIFT_FIXPT_SCALE = 48;
