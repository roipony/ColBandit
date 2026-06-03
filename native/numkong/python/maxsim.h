/**
 *  @brief MaxSim late-interaction declarations for NumKong Python bindings.
 *  @file python/maxsim.h
 *  @author Ash Vardanian
 *  @date March 9, 2026
 *
 *  Declares the MaxSimPackedMatrix type and API functions for MaxSim
 *  (ColBERT late-interaction scoring) used by the Python module.
 */
#ifndef NK_PYTHON_MAXSIM_H
#define NK_PYTHON_MAXSIM_H

#include "numkong.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Pre-packed matrix for MaxSim late-interaction scoring. */
typedef struct MaxSimPackedMatrix {
    PyObject_HEAD nk_dtype_t dtype;
    nk_size_t vector_count;
    nk_size_t depth;
    char start[];
} MaxSimPackedMatrix;

extern PyTypeObject MaxSimPackedMatrixType;

PyObject *api_maxsim_pack(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_colbandit_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_topm_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_topm_flat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim_pack_set_indices(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_maxsim_pack_set_4bit(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_colbandit_flat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_extract_flat_from_packed(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_total_tokens(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_full_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_centroid_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_centroid4_maxsim(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

extern char const doc_maxsim_pack[];
extern char const doc_maxsim_packed[];
extern char const doc_maxsim[];
extern char const doc_colbandit_maxsim[];
extern char const doc_topm_maxsim[];
extern char const doc_topm_flat[];
extern char const doc_maxsim_pack_set_indices[];
extern char const doc_maxsim_pack_set_4bit[];
extern char const doc_colbandit_flat[];
extern char const doc_extract_flat_from_packed[];
extern char const doc_total_tokens[];
extern char const doc_full_maxsim[];
extern char const doc_centroid_maxsim[];
extern char const doc_centroid4_maxsim[];
extern char const doc_score_4bit_compact[];
extern char const doc_score_i8_compact[];

PyObject *api_score_4bit_compact(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);
PyObject *api_score_i8_compact(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames);

#ifdef __cplusplus
}
#endif

#endif // NK_PYTHON_MAXSIM_H
