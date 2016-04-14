Abstract Requirements
===========================

* **Compartments with special properties**
  -  (e.g. capacitance) will be modeled as "subclasses"
  - "Subclass" means a Compartment that inherits another Compartment, as is
    interchangeable with its parent in all other situations where its parent
    would normally be required.
  - Calcium or other ionic compartments are functionally identical (insofar as
    Myriad's) object system is concerned.
     
* **Section (Abstraction) Implementation**
   - Sections represent collections of Compartments that concretely linked via
     adjacency mechs:
     `
     class Section(object):
    
     `

Concrete Requirements
===========================

1. Finish CU compilation
2. Re-write Dockerfile
3. Keep better track of what compartments are connected to which mechanisms
4. 
