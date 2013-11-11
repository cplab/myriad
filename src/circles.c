#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include "Object.h"
#include "Point.h"
#include "Circle.h"

int main (int argc, char ** argv)
{	void * p;

	initCircle();

	while (* ++ argv)
	{	switch (** argv) {
		case 'c':
			p = new(Circle, 1, 2, 3);
			break;
		case 'p':
			p = new(Point, 1, 2);
			break;
		default:
			continue;
		}

		if (isOf(p, Circle))
		{
			puts("P is of Circle");
		}

		if (isOf(p, Point))
		{
			puts("P is of Point.");
		}

		draw(p);
		move(p, 10, 20);
		draw(p);
		delete(p);
	}
	return 0;
}
