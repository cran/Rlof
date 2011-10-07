#ifndef DISTANCE_H
#define DISTANCE_H
extern "C" {
	void Rdistance(double *x, int *nr, int *nc, double *d, int *diag,
				   int *method, double *p);
}

#endif // DISTANCE_H
