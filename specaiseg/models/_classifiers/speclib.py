'''Module for managing spectral libraries'''
import json
import sys

import numpy as np
import yaml


class LibClusters():
    def __init__(self, names=None, assignments=None, filename=None):

        if names:
            if assignments is None:
                assignments = np.arange(len(names))
            self.cdict = bidict(zip(names, assignments))
        else:
            self.cdict = bidict()

        if filename is None:
            filename = 'lib_clusters.yml'
        self.filename = filename

    def dump(self, filename=None):
        '''Dump to json file with assignments as keys'''
        if filename is None:
            filename = self.filename

        dict_to_dump = dict(self.cdict.inverse)

        empty_keys = [k for k, v in dict_to_dump.items() if not v]
        for k in empty_keys:
            del dict_to_dump[k]

        # with open(filename, 'w') as file:
        #     yaml.safe_dump(dict_to_dump, file,default_style='""')

        with open(filename, 'w') as file:
            json.dump(dict_to_dump, file, indent=4)

    def load(self, filename):
        with open(filename) as f:
            inverse = json.load(f)
        self.from_dict(inverse)
        self.filename = filename

    def from_dict(self, inverse_dict):
        self.cdict = bidict()
        for k in inverse_dict:
            for n in inverse_dict[k]:
                self.cdict[n] = k
            self.cdict[k] = k

    def get_eq(self, name):
        return self.cdict.inverse[self.cdict[name]]

    def group(self, groups_list):
        for g in groups_list:
            for k in g:
                if k not in self.cdict:
                    self.cdict[k] = k
            # Get all the current assignments of members in the new group
            assigns = [self.cdict[k] for k in g]

            # Come up with new assignment
            try:
                # Try to use the shortest assignment name (string assignments)
                new_assign = min(assigns, key=len)
            except:
                # Otherwise, use the minimum assignment name (numerical assignments)
                new_assign = min(assigns)

            # Loop through all the old assignments
            for a in assigns:
                # Get all the members that match the assignment
                names = self.cdict.inverse[a]
                for n in names:
                    # Make the new assignment
                    self.cdict[n] = new_assign

    def to_cluster_matrix(self, names):
        # Add entries if they don't exist
        for n in names:
            if n not in self.cdict:
                self.cdict[n] = n
        # Determine all the cluster names
        cluster_names = list(self.cdict.inverse.keys())

        # Create a dictionary to map cluster_name to position in list
        c2num = dict(zip(cluster_names, range(len(cluster_names))))

        # Initialize the cluster matrix
        M = np.zeros((len(names), len(cluster_names)), dtype=bool)

        for i, n in enumerate(names):
            # Look up the cluter name for each name, and then look up the index, and then assign the matrix value
            M[i, c2num[self.cdict[n]]] = True

        return M, names, cluster_names

        # M = np.zeros((len(names),len(names)))
        # covered = []
        # for i,n in enumerate(names):
        #     if n not in covered:
        #         eq = self.get_eq(n)
        #         for eq1 in eq:
        #             M[:,i]=np.logical_or(M[i,:],names==eq1)
        #         covered += eq
        # cluster_indices = np.any(M,axis=0)
        # M=M[:,cluster_indices]
        # cluster_names = names[cluster_indices]
        # return M,names,cluster_names

    def map_integer_labels(self, integer_labels, names):
        for n in names:
            if n not in self.cdict:
                self.cdict[n] = n
        M, names, cluster_names = self.to_cluster_matrix(names)
        map = np.argmax(M, axis=1)
        new_integer_labels = map[integer_labels]
        return new_integer_labels, cluster_names

    def make_long_names(self, cluster_names):
        return ['||'.join(self.get_eq(n)) for n in cluster_names]


def libclusters_from_matrix(M, all_names, cluster_names=None):
    if cluster_names is not None:
        assignments = list()
        for i in range(len(all_names)):
            match = np.argmax(M[i, :])
            assignments.append(all_names[match])
        return LibClusters(all_names, assignments=assignments)
    else:
        libc = LibClusters(names=all_names)
        all_names = np.asarray(all_names)
        for i in range(len(all_names)):
            libc.group([all_names[M[:, i]]])
        return libc


def libclusters_from_file(filename):
    libc = LibClusters()
    libc.load(filename)
    return libc


def interactive_downselect(candidates):
    i = 0
    keep = np.zeros(len(candidates), dtype=bool)
    while i < len(candidates):
        print(candidates[i])
        uinput = input()
        if uinput == 'y' or uinput == 'Y':
            keep[i] = True
            i += 1
        elif uinput == 'n' or uinput == 'N':
            keep[i] = False
            i += 1
        elif uinput == 'u' or uinput == 'U':
            if i > 0:
                i -= 1
        elif uinput == 'q' or uinput == 'Q':
            break
    return np.asarray(candidates)[keep]


class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
