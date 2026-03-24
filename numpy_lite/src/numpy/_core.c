/* numpy._core — minimal NumPy replacement C extension */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <stdint.h>

#define NPL_MAXDIM 32

/* ── Section 1: Dtype enum & tables ──────────────────────────────────── */

enum {
    DTYPE_BOOL = 0,
    DTYPE_INT8, DTYPE_INT16, DTYPE_INT32, DTYPE_INT64,
    DTYPE_UINT8, DTYPE_UINT16, DTYPE_UINT32,
    DTYPE_FLOAT16, DTYPE_FLOAT32, DTYPE_FLOAT64,
    DTYPE_COUNT
};

static const char *dtype_names[DTYPE_COUNT] = {
    "bool_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32",
    "float16", "float32", "float64"
};

static const int dtype_sizes[DTYPE_COUNT] = {
    1, 1, 2, 4, 8,
    1, 2, 4,
    2, 4, 8
};

/* ── Section 2: Float16 conversion ───────────────────────────────────── */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static uint16_t f32_to_f16(float v) {
    uint32_t f;
    memcpy(&f, &v, 4);
    uint16_t sign = (uint16_t)((f >> 16) & 0x8000);
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(1 - exp + 13);
        if (shift < 32) mant >>= shift; else mant = 0;
        return sign | (uint16_t)(mant);
    }
    if (exp == 0xFF - 127 + 15) {
        if (mant) return sign | 0x7E00;
        return sign | 0x7C00;
    }
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)((uint32_t)exp << 10) | (uint16_t)(mant >> 13);
}

/* ── Section 3: Type forward declarations ────────────────────────────── */

typedef struct { PyObject_HEAD int dtype_enum; } DtypeObject;
static PyTypeObject DtypeType;
static DtypeObject *dtype_singletons[DTYPE_COUNT];

typedef struct {
    PyObject_HEAD
    char *data;
    Py_ssize_t shape[NPL_MAXDIM];
    Py_ssize_t strides[NPL_MAXDIM];
    int ndim;
    int dtype;
    Py_ssize_t size;
    PyObject *base;
} NdArrayObject;
static PyTypeObject NdArrayType;

/* Forward declarations for methods referenced before definition */
static PyObject *NdArray_format(NdArrayObject *self, PyObject *args);

/* ── Section 4: DtypeObject implementation ───────────────────────────── */

static void Dtype_dealloc(DtypeObject *self) { (void)self; }

static PyObject *Dtype_repr(DtypeObject *self) {
    const char *n = dtype_names[self->dtype_enum];
    if (self->dtype_enum == DTYPE_BOOL)
        return PyUnicode_FromFormat("dtype('bool')");
    return PyUnicode_FromFormat("dtype('%s')", n);
}

static PyObject *Dtype_richcompare(PyObject *a, PyObject *b, int op) {
    if (!PyObject_TypeCheck(a, &DtypeType) || !PyObject_TypeCheck(b, &DtypeType))
        Py_RETURN_NOTIMPLEMENTED;
    int ea = ((DtypeObject*)a)->dtype_enum;
    int eb = ((DtypeObject*)b)->dtype_enum;
    int res;
    if (op == Py_EQ) res = (ea == eb);
    else if (op == Py_NE) res = (ea != eb);
    else Py_RETURN_NOTIMPLEMENTED;
    return PyBool_FromLong(res);
}

static PyObject *Dtype_get_itemsize(DtypeObject *self, void *c) {
    (void)c;
    return PyLong_FromLong(dtype_sizes[self->dtype_enum]);
}

static PyObject *Dtype_get_name(DtypeObject *self, void *c) {
    (void)c;
    if (self->dtype_enum == DTYPE_BOOL) return PyUnicode_FromString("bool");
    return PyUnicode_FromString(dtype_names[self->dtype_enum]);
}

static Py_hash_t Dtype_hash(DtypeObject *self) {
    return (Py_hash_t)(self->dtype_enum + 1);
}

static PyObject *Dtype_get_type(DtypeObject *self, void *c) {
    (void)c;
    Py_INCREF(dtype_singletons[self->dtype_enum]);
    return (PyObject*)dtype_singletons[self->dtype_enum];
}

static int _parse_dtype_enum(PyObject *arg) {
    if (PyObject_TypeCheck(arg, &DtypeType))
        return ((DtypeObject*)arg)->dtype_enum;
    if (PyUnicode_Check(arg)) {
        const char *s = PyUnicode_AsUTF8(arg);
        if (!s) return -1;
        for (int i = 0; i < DTYPE_COUNT; i++)
            if (strcmp(s, dtype_names[i]) == 0) return i;
        if (strcmp(s, "bool") == 0) return DTYPE_BOOL;
        if (strcmp(s, "float") == 0 || strcmp(s, "float64") == 0 || strcmp(s, "double") == 0)
            return DTYPE_FLOAT64;
        if (strcmp(s, "int") == 0 || strcmp(s, "int64") == 0 || strcmp(s, "long") == 0)
            return DTYPE_INT64;
        PyErr_Format(PyExc_TypeError, "Unknown dtype: '%s'", s);
        return -1;
    }
    if (arg == (PyObject*)&PyFloat_Type) return DTYPE_FLOAT64;
    if (arg == (PyObject*)&PyLong_Type)  return DTYPE_INT64;
    if (arg == (PyObject*)&PyBool_Type)  return DTYPE_BOOL;
    PyErr_SetString(PyExc_TypeError, "Cannot interpret as dtype");
    return -1;
}

static PyObject *Dtype_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    (void)type; (void)kw;
    PyObject *arg;
    if (!PyArg_ParseTuple(args, "O", &arg)) return NULL;
    if (PyObject_TypeCheck(arg, &DtypeType)) { Py_INCREF(arg); return arg; }
    int e = _parse_dtype_enum(arg);
    if (e < 0) return NULL;
    Py_INCREF(dtype_singletons[e]);
    return (PyObject*)dtype_singletons[e];
}

static PyGetSetDef Dtype_getset[] = {
    {"itemsize", (getter)Dtype_get_itemsize, NULL, NULL, NULL},
    {"name",     (getter)Dtype_get_name,     NULL, NULL, NULL},
    {"type",     (getter)Dtype_get_type,     NULL, NULL, NULL},
    {NULL}
};

static PyTypeObject DtypeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "numpy.dtype",
    .tp_basicsize = sizeof(DtypeObject),
    .tp_dealloc   = (destructor)Dtype_dealloc,
    .tp_repr      = (reprfunc)Dtype_repr,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_hash      = (hashfunc)Dtype_hash,
    .tp_richcompare = Dtype_richcompare,
    .tp_getset    = Dtype_getset,
    .tp_new       = Dtype_new,
};

/* ── Section 5: Value access helpers ─────────────────────────────────── */

static double get_element(const char *data, int dtype, Py_ssize_t off) {
    const char *p = data + off;
    switch (dtype) {
        case DTYPE_BOOL:    return (double)(*(const uint8_t*)p != 0);
        case DTYPE_INT8:    return (double)(*(const int8_t*)p);
        case DTYPE_INT16:   return (double)(*(const int16_t*)p);
        case DTYPE_INT32:   return (double)(*(const int32_t*)p);
        case DTYPE_INT64:   return (double)(*(const int64_t*)p);
        case DTYPE_UINT8:   return (double)(*(const uint8_t*)p);
        case DTYPE_UINT16:  return (double)(*(const uint16_t*)p);
        case DTYPE_UINT32:  return (double)(*(const uint32_t*)p);
        case DTYPE_FLOAT16: return (double)f16_to_f32(*(const uint16_t*)p);
        case DTYPE_FLOAT32: return (double)(*(const float*)p);
        case DTYPE_FLOAT64: return *(const double*)p;
    }
    return 0.0;
}

static void set_element(char *data, int dtype, Py_ssize_t off, double val) {
    char *p = data + off;
    switch (dtype) {
        case DTYPE_BOOL:    *(uint8_t*)p  = (val != 0.0) ? 1 : 0; break;
        case DTYPE_INT8:    *(int8_t*)p   = (int8_t)val;   break;
        case DTYPE_INT16:   *(int16_t*)p  = (int16_t)val;  break;
        case DTYPE_INT32:   *(int32_t*)p  = (int32_t)val;  break;
        case DTYPE_INT64:   *(int64_t*)p  = (int64_t)val;  break;
        case DTYPE_UINT8:   *(uint8_t*)p  = (uint8_t)val;  break;
        case DTYPE_UINT16:  *(uint16_t*)p = (uint16_t)val; break;
        case DTYPE_UINT32:  *(uint32_t*)p = (uint32_t)val; break;
        case DTYPE_FLOAT16: *(uint16_t*)p = f32_to_f16((float)val); break;
        case DTYPE_FLOAT32: *(float*)p    = (float)val;    break;
        case DTYPE_FLOAT64: *(double*)p   = val;           break;
    }
}

/* ── Section 6: NdArray allocation ───────────────────────────────────── */

static Py_ssize_t compute_size(const Py_ssize_t *shape, int ndim) {
    Py_ssize_t s = 1;
    for (int i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

static void compute_strides(NdArrayObject *a) {
    if (a->ndim == 0) return;
    int es = dtype_sizes[a->dtype];
    a->strides[a->ndim - 1] = es;
    for (int i = a->ndim - 2; i >= 0; i--)
        a->strides[i] = a->strides[i + 1] * a->shape[i + 1];
}

static NdArrayObject *ndarray_new_empty(int dtype, const Py_ssize_t *shape, int ndim) {
    NdArrayObject *a = PyObject_New(NdArrayObject, &NdArrayType);
    if (!a) return NULL;
    a->ndim = ndim;
    a->dtype = dtype;
    a->base = NULL;
    for (int i = 0; i < ndim; i++) a->shape[i] = shape[i];
    a->size = compute_size(shape, ndim);
    compute_strides(a);
    Py_ssize_t nbytes = a->size * dtype_sizes[dtype];
    if (nbytes < 1) nbytes = dtype_sizes[dtype];
    a->data = (char*)calloc(1, (size_t)nbytes);
    if (!a->data) { Py_DECREF(a); PyErr_NoMemory(); return NULL; }
    return a;
}

static void NdArray_dealloc(NdArrayObject *self) {
    if (!self->base) free(self->data);
    Py_XDECREF(self->base);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* ── Section 7: Core helpers ─────────────────────────────────────────── */

static int is_c_contiguous(NdArrayObject *a) {
    if (a->ndim == 0) return 1;
    int es = dtype_sizes[a->dtype];
    Py_ssize_t expected = es;
    for (int i = a->ndim - 1; i >= 0; i--) {
        if (a->shape[i] == 0) return 1;
        if (a->shape[i] != 1 && a->strides[i] != expected) return 0;
        expected *= a->shape[i];
    }
    return 1;
}

static int promote_dtype(int a, int b) {
    if (a == b) return a;
    int af = (a >= DTYPE_FLOAT16), bf = (b >= DTYPE_FLOAT16);
    if (af || bf) {
        int fa = af ? a : DTYPE_FLOAT64;
        int fb = bf ? b : DTYPE_FLOAT64;
        if (fa == DTYPE_FLOAT64 || fb == DTYPE_FLOAT64) return DTYPE_FLOAT64;
        if (fa == DTYPE_FLOAT32 || fb == DTYPE_FLOAT32) return DTYPE_FLOAT32;
        return DTYPE_FLOAT16;
    }
    int sa = dtype_sizes[a], sb = dtype_sizes[b];
    if (sa >= sb) return a;
    return b;
}

static int broadcast_shapes(const Py_ssize_t *a, int an, const Py_ssize_t *b, int bn,
                            Py_ssize_t *out, int *out_ndim) {
    int n = an > bn ? an : bn;
    *out_ndim = n;
    for (int i = 0; i < n; i++) {
        Py_ssize_t da = (i < n - an) ? 1 : a[i - (n - an)];
        Py_ssize_t db = (i < n - bn) ? 1 : b[i - (n - bn)];
        if (da == db) out[i] = da;
        else if (da == 1) out[i] = db;
        else if (db == 1) out[i] = da;
        else {
            PyErr_Format(PyExc_ValueError,
                "operands could not be broadcast together with shapes");
            return -1;
        }
    }
    return 0;
}

static void broadcast_strides(const Py_ssize_t *shape, const Py_ssize_t *strides,
                              int ndim, int out_ndim, Py_ssize_t *out_strides) {
    int delta = out_ndim - ndim;
    for (int i = 0; i < out_ndim; i++) {
        int j = i - delta;
        if (j < 0 || shape[j] == 1) out_strides[i] = 0;
        else out_strides[i] = strides[j];
    }
}

static NdArrayObject *to_ndarray(PyObject *obj) {
    if (PyObject_TypeCheck(obj, &NdArrayType)) {
        Py_INCREF(obj);
        return (NdArrayObject*)obj;
    }
    NdArrayObject *a;
    if (PyBool_Check(obj)) {
        a = ndarray_new_empty(DTYPE_BOOL, NULL, 0);
        if (!a) return NULL;
        set_element(a->data, DTYPE_BOOL, 0, obj == Py_True ? 1.0 : 0.0);
        return a;
    }
    if (PyLong_Check(obj)) {
        a = ndarray_new_empty(DTYPE_INT64, NULL, 0);
        if (!a) return NULL;
        set_element(a->data, DTYPE_INT64, 0, (double)PyLong_AsLongLong(obj));
        return a;
    }
    if (PyFloat_Check(obj)) {
        a = ndarray_new_empty(DTYPE_FLOAT64, NULL, 0);
        if (!a) return NULL;
        set_element(a->data, DTYPE_FLOAT64, 0, PyFloat_AsDouble(obj));
        return a;
    }
    return NULL;
}

static NdArrayObject *make_contiguous_copy(NdArrayObject *src) {
    NdArrayObject *dst = ndarray_new_empty(src->dtype, src->shape, src->ndim);
    if (!dst) return NULL;
    int es = dtype_sizes[src->dtype];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < src->size; i++) {
        Py_ssize_t src_off = 0;
        for (int d = 0; d < src->ndim; d++) src_off += idx[d] * src->strides[d];
        double v = get_element(src->data, src->dtype, src_off);
        set_element(dst->data, dst->dtype, i * es, v);
        for (int d = src->ndim - 1; d >= 0; d--) {
            if (++idx[d] < src->shape[d]) break;
            idx[d] = 0;
        }
    }
    return dst;
}

/* ── Section 8: Unary operation framework ────────────────────────────── */

typedef double (*unary_fn)(double);

static NdArrayObject *unary_op(NdArrayObject *a, unary_fn fn, int out_dtype) {
    NdArrayObject *r = ndarray_new_empty(out_dtype, a->shape, a->ndim);
    if (!r) return NULL;
    int oes = dtype_sizes[out_dtype];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < a->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < a->ndim; d++) off += idx[d] * a->strides[d];
        set_element(r->data, out_dtype, i * oes, fn(get_element(a->data, a->dtype, off)));
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }
    return r;
}

static double _op_neg(double x) { return -x; }
static double _op_abs(double x) { return fabs(x); }
static double _op_exp(double x) { return exp(x); }
static double _op_log(double x) { return log(x); }
static double _op_sqrt(double x) { return sqrt(x); }
static double _op_tanh(double x) { return tanh(x); }
static double _op_isnan_d(double x) { return isnan(x) ? 1.0 : 0.0; }
static double _op_isinf_d(double x) { return isinf(x) ? 1.0 : 0.0; }
static double _op_isposinf_d(double x) { return (isinf(x) && x > 0) ? 1.0 : 0.0; }
static double _op_isneginf_d(double x) { return (isinf(x) && x < 0) ? 1.0 : 0.0; }
static double _op_logical_not_d(double x) { return x == 0.0 ? 1.0 : 0.0; }

/* ── Section 9: Binary operation framework ───────────────────────────── */

typedef double (*binary_fn)(double, double);

static NdArrayObject *binary_op(NdArrayObject *a, NdArrayObject *b,
                                binary_fn fn, int out_dtype) {
    Py_ssize_t out_shape[NPL_MAXDIM];
    int out_ndim;
    if (broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim,
                         out_shape, &out_ndim) < 0) return NULL;
    NdArrayObject *r = ndarray_new_empty(out_dtype, out_shape, out_ndim);
    if (!r) return NULL;
    Py_ssize_t sa[NPL_MAXDIM], sb[NPL_MAXDIM];
    broadcast_strides(a->shape, a->strides, a->ndim, out_ndim, sa);
    broadcast_strides(b->shape, b->strides, b->ndim, out_ndim, sb);
    int oes = dtype_sizes[out_dtype];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    Py_ssize_t total = compute_size(out_shape, out_ndim);
    for (Py_ssize_t i = 0; i < total; i++) {
        Py_ssize_t oa = 0, ob = 0;
        for (int d = 0; d < out_ndim; d++) { oa += idx[d]*sa[d]; ob += idx[d]*sb[d]; }
        double va = get_element(a->data, a->dtype, oa);
        double vb = get_element(b->data, b->dtype, ob);
        set_element(r->data, out_dtype, i * oes, fn(va, vb));
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return r;
}

static double _op_add(double a, double b) { return a + b; }
static double _op_sub(double a, double b) { return a - b; }
static double _op_mul(double a, double b) { return a * b; }
static double _op_div(double a, double b) { return a / b; }
static double _op_maximum(double a, double b) { return a >= b ? a : b; }
static double _op_minimum(double a, double b) { return a <= b ? a : b; }
static double _op_eq(double a, double b) { return a == b ? 1.0 : 0.0; }
static double _op_ne(double a, double b) { return a != b ? 1.0 : 0.0; }
static double _op_lt(double a, double b) { return a < b ? 1.0 : 0.0; }
static double _op_le(double a, double b) { return a <= b ? 1.0 : 0.0; }
static double _op_gt(double a, double b) { return a > b ? 1.0 : 0.0; }
static double _op_ge(double a, double b) { return a >= b ? 1.0 : 0.0; }
static double _op_logical_and_d(double a, double b) { return (a && b) ? 1.0 : 0.0; }
static double _op_logical_or_d(double a, double b) { return (a || b) ? 1.0 : 0.0; }
static double _op_logical_xor_d(double a, double b) { return (!a != !b) ? 1.0 : 0.0; }

/* ── Section 10: Reduction framework ────────────────────────────────── */

static double _comb_add(double a, double b) { return a + b; }
static double _comb_max(double a, double b) { return a > b ? a : b; }
static double _comb_min(double a, double b) { return a < b ? a : b; }

#define AXIS_NONE (-999)

static NdArrayObject *reduce_op(NdArrayObject *self, int axis, int keepdims,
                                double init_val, double (*combine)(double, double),
                                int do_mean) {
    if (axis == AXIS_NONE) {
        Py_ssize_t out_shape[NPL_MAXDIM];
        int out_ndim;
        if (keepdims) {
            out_ndim = self->ndim;
            for (int i = 0; i < out_ndim; i++) out_shape[i] = 1;
        } else {
            out_ndim = 0;
        }
        NdArrayObject *r = ndarray_new_empty(self->dtype, out_shape, out_ndim);
        if (!r) return NULL;
        double acc = init_val;
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < self->size; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
            acc = combine(acc, get_element(self->data, self->dtype, off));
            for (int d = self->ndim - 1; d >= 0; d--) {
                if (++idx[d] < self->shape[d]) break;
                idx[d] = 0;
            }
        }
        if (do_mean && self->size > 0) acc /= (double)self->size;
        set_element(r->data, r->dtype, 0, acc);
        return r;
    }
    if (axis < 0) axis += self->ndim;
    if (axis < 0 || axis >= self->ndim) {
        PyErr_SetString(PyExc_ValueError, "axis out of range");
        return NULL;
    }
    Py_ssize_t out_shape[NPL_MAXDIM];
    int out_ndim = 0;
    for (int i = 0; i < self->ndim; i++) {
        if (i == axis) {
            if (keepdims) out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = self->shape[i];
        }
    }
    NdArrayObject *r = ndarray_new_empty(self->dtype, out_shape, out_ndim);
    if (!r) return NULL;
    Py_ssize_t out_size = compute_size(out_shape, out_ndim);
    int oes = dtype_sizes[self->dtype];
    Py_ssize_t oidx[NPL_MAXDIM];
    memset(oidx, 0, sizeof(oidx));
    for (Py_ssize_t oi = 0; oi < out_size; oi++) {
        Py_ssize_t src_idx[NPL_MAXDIM];
        int si = 0;
        for (int d = 0; d < self->ndim; d++) {
            if (d == axis) { src_idx[d] = 0; }
            else {
                int od = si++;
                if (keepdims && d > axis) od = si - 1;
                src_idx[d] = oidx[od];
            }
        }
        /* Re-map: for keepdims, out has same rank with 1 at axis.
           For !keepdims, out has ndim-1 dims.
           Map oidx back to src_idx excluding axis. */
        {
            int oi2 = 0;
            for (int d = 0; d < self->ndim; d++) {
                if (d == axis) src_idx[d] = 0;
                else src_idx[d] = oidx[oi2++];
            }
        }
        double acc = init_val;
        for (Py_ssize_t k = 0; k < self->shape[axis]; k++) {
            src_idx[axis] = k;
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += src_idx[d]*self->strides[d];
            acc = combine(acc, get_element(self->data, self->dtype, off));
        }
        if (do_mean && self->shape[axis] > 0) acc /= (double)self->shape[axis];
        set_element(r->data, r->dtype, oi * oes, acc);
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++oidx[d] < out_shape[d]) break;
            oidx[d] = 0;
        }
    }
    return r;
}

static NdArrayObject *reduce_arg_op(NdArrayObject *self, int axis, int find_max) {
    if (axis == AXIS_NONE) {
        NdArrayObject *r = ndarray_new_empty(DTYPE_INT64, NULL, 0);
        if (!r) return NULL;
        double best = get_element(self->data, self->dtype, 0);
        Py_ssize_t best_idx = 0;
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < self->size; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
            double v = get_element(self->data, self->dtype, off);
            if ((find_max && v > best) || (!find_max && v < best)) { best = v; best_idx = i; }
            for (int d = self->ndim - 1; d >= 0; d--) {
                if (++idx[d] < self->shape[d]) break;
                idx[d] = 0;
            }
        }
        set_element(r->data, DTYPE_INT64, 0, (double)best_idx);
        return r;
    }
    if (axis < 0) axis += self->ndim;
    if (axis < 0 || axis >= self->ndim) {
        PyErr_SetString(PyExc_ValueError, "axis out of range");
        return NULL;
    }
    Py_ssize_t out_shape[NPL_MAXDIM];
    int out_ndim = 0;
    for (int i = 0; i < self->ndim; i++)
        if (i != axis) out_shape[out_ndim++] = self->shape[i];
    NdArrayObject *r = ndarray_new_empty(DTYPE_INT64, out_shape, out_ndim);
    if (!r) return NULL;
    Py_ssize_t out_size = compute_size(out_shape, out_ndim);
    int oes = dtype_sizes[DTYPE_INT64];
    Py_ssize_t oidx[NPL_MAXDIM];
    memset(oidx, 0, sizeof(oidx));
    for (Py_ssize_t oi = 0; oi < out_size; oi++) {
        Py_ssize_t src_idx[NPL_MAXDIM];
        int oi2 = 0;
        for (int d = 0; d < self->ndim; d++) {
            if (d == axis) src_idx[d] = 0;
            else src_idx[d] = oidx[oi2++];
        }
        Py_ssize_t off0 = 0;
        for (int d = 0; d < self->ndim; d++) off0 += src_idx[d]*self->strides[d];
        double best = get_element(self->data, self->dtype, off0);
        Py_ssize_t best_k = 0;
        for (Py_ssize_t k = 1; k < self->shape[axis]; k++) {
            src_idx[axis] = k;
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += src_idx[d]*self->strides[d];
            double v = get_element(self->data, self->dtype, off);
            if ((find_max && v > best) || (!find_max && v < best)) { best = v; best_k = k; }
        }
        set_element(r->data, DTYPE_INT64, oi * oes, (double)best_k);
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++oidx[d] < out_shape[d]) break;
            oidx[d] = 0;
        }
    }
    return r;
}

static NdArrayObject *cumsum_op(NdArrayObject *self, int axis) {
    if (axis == AXIS_NONE) {
        Py_ssize_t n = self->size;
        NdArrayObject *r = ndarray_new_empty(self->dtype, &n, 1);
        if (!r) return NULL;
        int es = dtype_sizes[self->dtype];
        double acc = 0;
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < n; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
            acc += get_element(self->data, self->dtype, off);
            set_element(r->data, r->dtype, i * es, acc);
            for (int d = self->ndim - 1; d >= 0; d--) {
                if (++idx[d] < self->shape[d]) break;
                idx[d] = 0;
            }
        }
        return r;
    }
    if (axis < 0) axis += self->ndim;
    if (axis < 0 || axis >= self->ndim) {
        PyErr_SetString(PyExc_ValueError, "axis out of range");
        return NULL;
    }
    NdArrayObject *r = ndarray_new_empty(self->dtype, self->shape, self->ndim);
    if (!r) return NULL;
    Py_ssize_t oidx[NPL_MAXDIM];
    memset(oidx, 0, sizeof(oidx));
    for (Py_ssize_t i = 0; i < r->size; i++) {
        /* We iterate in C order. For cumsum along axis, we accumulate. */
        Py_ssize_t src_off = 0, dst_off = 0;
        for (int d = 0; d < self->ndim; d++) src_off += oidx[d]*self->strides[d];
        for (int d = 0; d < r->ndim; d++) dst_off += oidx[d]*r->strides[d];
        double v = get_element(self->data, self->dtype, src_off);
        if (oidx[axis] > 0) {
            Py_ssize_t prev_idx[NPL_MAXDIM];
            memcpy(prev_idx, oidx, sizeof(oidx));
            prev_idx[axis]--;
            Py_ssize_t prev_off = 0;
            for (int d = 0; d < r->ndim; d++) prev_off += prev_idx[d]*r->strides[d];
            v += get_element(r->data, r->dtype, prev_off);
        }
        set_element(r->data, r->dtype, dst_off, v);
        for (int d = r->ndim - 1; d >= 0; d--) {
            if (++oidx[d] < r->shape[d]) break;
            oidx[d] = 0;
        }
    }
    return r;
}

static NdArrayObject *var_op(NdArrayObject *self, int axis, int keepdims) {
    NdArrayObject *m = reduce_op(self, axis, 1, 0.0, _comb_add, 1);
    if (!m) return NULL;
    /* Two-pass: compute mean, then variance */
    int out_dtype = self->dtype;
    if (axis == AXIS_NONE) {
        Py_ssize_t out_shape[NPL_MAXDIM];
        int out_ndim;
        if (keepdims) {
            out_ndim = self->ndim;
            for (int i = 0; i < out_ndim; i++) out_shape[i] = 1;
        } else {
            out_ndim = 0;
        }
        NdArrayObject *r = ndarray_new_empty(out_dtype, out_shape, out_ndim);
        if (!r) { Py_DECREF(m); return NULL; }
        double mean_val = get_element(m->data, m->dtype, 0);
        double acc = 0;
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < self->size; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
            double diff = get_element(self->data, self->dtype, off) - mean_val;
            acc += diff * diff;
            for (int d = self->ndim - 1; d >= 0; d--) {
                if (++idx[d] < self->shape[d]) break;
                idx[d] = 0;
            }
        }
        if (self->size > 0) acc /= (double)self->size;
        set_element(r->data, r->dtype, 0, acc);
        Py_DECREF(m);
        return r;
    }
    int ax = axis;
    if (ax < 0) ax += self->ndim;
    Py_ssize_t out_shape[NPL_MAXDIM];
    int out_ndim = 0;
    for (int i = 0; i < self->ndim; i++) {
        if (i == ax) { if (keepdims) out_shape[out_ndim++] = 1; }
        else out_shape[out_ndim++] = self->shape[i];
    }
    NdArrayObject *r = ndarray_new_empty(out_dtype, out_shape, out_ndim);
    if (!r) { Py_DECREF(m); return NULL; }
    Py_ssize_t out_size = compute_size(out_shape, out_ndim);
    int oes = dtype_sizes[out_dtype];
    Py_ssize_t oidx[NPL_MAXDIM];
    memset(oidx, 0, sizeof(oidx));
    for (Py_ssize_t oi = 0; oi < out_size; oi++) {
        Py_ssize_t src_idx[NPL_MAXDIM], m_idx[NPL_MAXDIM];
        int oi2 = 0;
        for (int d = 0; d < self->ndim; d++) {
            if (d == ax) { src_idx[d] = 0; m_idx[d] = 0; }
            else { src_idx[d] = oidx[oi2]; m_idx[d] = oidx[oi2]; oi2++; }
        }
        /* Get mean for this output position */
        Py_ssize_t m_off = 0;
        for (int d = 0; d < m->ndim; d++) m_off += m_idx[d]*m->strides[d];
        double mean_val = get_element(m->data, m->dtype, m_off);
        double acc = 0;
        for (Py_ssize_t k = 0; k < self->shape[ax]; k++) {
            src_idx[ax] = k;
            Py_ssize_t off = 0;
            for (int d = 0; d < self->ndim; d++) off += src_idx[d]*self->strides[d];
            double diff = get_element(self->data, self->dtype, off) - mean_val;
            acc += diff * diff;
        }
        if (self->shape[ax] > 0) acc /= (double)self->shape[ax];
        set_element(r->data, r->dtype, oi * oes, acc);
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++oidx[d] < out_shape[d]) break;
            oidx[d] = 0;
        }
    }
    Py_DECREF(m);
    return r;
}

/* ── Section 11: Array construction from Python objects ──────────────── */

static int _infer_shape(PyObject *obj, Py_ssize_t *shape, int *ndim) {
    *ndim = 0;
    PyObject *cur = obj;
    Py_INCREF(cur);
    while (PyList_Check(cur) || PyTuple_Check(cur)) {
        Py_ssize_t n = PySequence_Size(cur);
        if (n < 0) { Py_DECREF(cur); return -1; }
        if (*ndim >= NPL_MAXDIM) { Py_DECREF(cur); PyErr_SetString(PyExc_ValueError, "too many dims"); return -1; }
        shape[*ndim] = n;
        (*ndim)++;
        if (n == 0) { Py_DECREF(cur); return 0; }
        PyObject *first = PySequence_GetItem(cur, 0);
        Py_DECREF(cur);
        cur = first;
        if (!cur) return -1;
    }
    /* cur is a scalar or ndarray */
    if (PyObject_TypeCheck(cur, &NdArrayType)) {
        NdArrayObject *a = (NdArrayObject*)cur;
        for (int i = 0; i < a->ndim; i++) {
            if (*ndim >= NPL_MAXDIM) { Py_DECREF(cur); return -1; }
            shape[*ndim] = a->shape[i];
            (*ndim)++;
        }
    }
    Py_DECREF(cur);
    return 0;
}

static int _infer_dtype(PyObject *obj, int depth, int ndim) {
    if (depth >= ndim) {
        if (PyBool_Check(obj)) return DTYPE_BOOL;
        if (PyLong_Check(obj)) return DTYPE_INT64;
        if (PyFloat_Check(obj)) return DTYPE_FLOAT64;
        if (PyObject_TypeCheck(obj, &NdArrayType))
            return ((NdArrayObject*)obj)->dtype;
        return DTYPE_FLOAT64;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        if (PyBool_Check(obj)) return DTYPE_BOOL;
        if (PyLong_Check(obj)) return DTYPE_INT64;
        if (PyFloat_Check(obj)) return DTYPE_FLOAT64;
        return DTYPE_FLOAT64;
    }
    Py_ssize_t n = PySequence_Size(obj);
    int dt = DTYPE_BOOL;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) return -1;
        int sub = _infer_dtype(item, depth + 1, ndim);
        Py_DECREF(item);
        if (sub < 0) return -1;
        dt = promote_dtype(dt, sub);
    }
    return dt;
}

static int _fill_from_seq(NdArrayObject *arr, PyObject *obj, int depth, Py_ssize_t *flat_idx) {
    int es = dtype_sizes[arr->dtype];
    if (depth >= arr->ndim) {
        if (PyObject_TypeCheck(obj, &NdArrayType)) {
            NdArrayObject *src = (NdArrayObject*)obj;
            Py_ssize_t idx2[NPL_MAXDIM];
            memset(idx2, 0, sizeof(idx2));
            for (Py_ssize_t i = 0; i < src->size; i++) {
                Py_ssize_t off = 0;
                for (int d = 0; d < src->ndim; d++) off += idx2[d]*src->strides[d];
                double v = get_element(src->data, src->dtype, off);
                set_element(arr->data, arr->dtype, (*flat_idx) * es, v);
                (*flat_idx)++;
                for (int d = src->ndim - 1; d >= 0; d--) {
                    if (++idx2[d] < src->shape[d]) break;
                    idx2[d] = 0;
                }
            }
            return 0;
        }
        double v;
        if (PyBool_Check(obj)) v = (obj == Py_True) ? 1.0 : 0.0;
        else if (PyLong_Check(obj)) v = (double)PyLong_AsLongLong(obj);
        else if (PyFloat_Check(obj)) v = PyFloat_AsDouble(obj);
        else v = 0.0;
        set_element(arr->data, arr->dtype, (*flat_idx) * es, v);
        (*flat_idx)++;
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        double v;
        if (PyBool_Check(obj)) v = (obj == Py_True) ? 1.0 : 0.0;
        else if (PyLong_Check(obj)) v = (double)PyLong_AsLongLong(obj);
        else if (PyFloat_Check(obj)) v = PyFloat_AsDouble(obj);
        else v = 0.0;
        set_element(arr->data, arr->dtype, (*flat_idx) * es, v);
        (*flat_idx)++;
        return 0;
    }
    Py_ssize_t n = PySequence_Size(obj);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(obj, i);
        if (!item) return -1;
        if (_fill_from_seq(arr, item, depth + 1, flat_idx) < 0) {
            Py_DECREF(item);
            return -1;
        }
        Py_DECREF(item);
    }
    return 0;
}

/* ── Section 12: NdArray methods ─────────────────────────────────────── */

static int _parse_axis_keepdims(PyObject *args, PyObject *kwargs, int *axis, int *keepdims) {
    *axis = AXIS_NONE;
    *keepdims = 0;
    static char *kwlist[] = {"axis", "keepdims", NULL};
    PyObject *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Op", kwlist, &axis_obj, keepdims))
        return -1;
    if (axis_obj != Py_None) {
        *axis = (int)PyLong_AsLong(axis_obj);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

static PyObject *NdArray_astype(NdArrayObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"dtype", "copy", NULL};
    PyObject *dt_obj;
    int copy = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &dt_obj, &copy))
        return NULL;
    int new_dt = _parse_dtype_enum(dt_obj);
    if (new_dt < 0) return NULL;
    if (!copy && new_dt == self->dtype) { Py_INCREF(self); return (PyObject*)self; }
    NdArrayObject *r = ndarray_new_empty(new_dt, self->shape, self->ndim);
    if (!r) return NULL;
    int oes = dtype_sizes[new_dt];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
        set_element(r->data, new_dt, i * oes, get_element(self->data, self->dtype, off));
        for (int d = self->ndim - 1; d >= 0; d--) {
            if (++idx[d] < self->shape[d]) break;
            idx[d] = 0;
        }
    }
    return (PyObject*)r;
}

static PyObject *NdArray_copy(NdArrayObject *self, PyObject *Py_UNUSED(ignored)) {
    return (PyObject*)make_contiguous_copy(self);
}

static int _parse_shape_from_args(PyObject *args, Py_ssize_t *shape, int *ndim) {
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs == 1) {
        PyObject *arg0 = PyTuple_GetItem(args, 0);
        if (PyTuple_Check(arg0) || PyList_Check(arg0)) {
            Py_ssize_t n = PySequence_Size(arg0);
            *ndim = (int)n;
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *item = PySequence_GetItem(arg0, i);
                shape[i] = PyLong_AsSsize_t(item);
                Py_DECREF(item);
                if (PyErr_Occurred()) return -1;
            }
            return 0;
        }
        if (PyLong_Check(arg0)) {
            *ndim = 1;
            shape[0] = PyLong_AsSsize_t(arg0);
            return 0;
        }
    }
    *ndim = (int)nargs;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        PyObject *item = PyTuple_GetItem(args, i);
        shape[i] = PyLong_AsSsize_t(item);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

static PyObject *NdArray_reshape(NdArrayObject *self, PyObject *args) {
    Py_ssize_t new_shape[NPL_MAXDIM];
    int new_ndim;
    if (_parse_shape_from_args(args, new_shape, &new_ndim) < 0) return NULL;

    /* Handle -1 */
    int neg_idx = -1;
    Py_ssize_t known = 1;
    for (int i = 0; i < new_ndim; i++) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0) { PyErr_SetString(PyExc_ValueError, "only one -1 allowed"); return NULL; }
            neg_idx = i;
        } else {
            known *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        if (known == 0) { PyErr_SetString(PyExc_ValueError, "cannot reshape"); return NULL; }
        new_shape[neg_idx] = self->size / known;
    }
    Py_ssize_t new_size = compute_size(new_shape, new_ndim);
    if (new_size != self->size) {
        PyErr_SetString(PyExc_ValueError, "cannot reshape array of different size");
        return NULL;
    }
    NdArrayObject *src = self;
    int need_copy = !is_c_contiguous(self);
    if (need_copy) {
        src = make_contiguous_copy(self);
        if (!src) return NULL;
    }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) { if (need_copy) Py_DECREF(src); return NULL; }
    r->ndim = new_ndim;
    r->dtype = src->dtype;
    r->data = src->data;
    r->size = new_size;
    memcpy(r->shape, new_shape, new_ndim * sizeof(Py_ssize_t));
    compute_strides(r);
    if (src->base) { r->base = src->base; Py_INCREF(r->base); }
    else { r->base = (PyObject*)src; Py_INCREF(src); }
    if (need_copy) Py_DECREF(src);
    return (PyObject*)r;
}

static PyObject *NdArray_item(NdArrayObject *self, PyObject *Py_UNUSED(ignored)) {
    double v = get_element(self->data, self->dtype, 0);
    if (self->dtype == DTYPE_BOOL) return PyBool_FromLong((long)v);
    if (self->dtype >= DTYPE_FLOAT16) return PyFloat_FromDouble(v);
    return PyLong_FromLongLong((long long)v);
}

static PyObject *NdArray_any(NdArrayObject *self, PyObject *Py_UNUSED(ignored)) {
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
        if (get_element(self->data, self->dtype, off) != 0.0) Py_RETURN_TRUE;
        for (int d = self->ndim - 1; d >= 0; d--) {
            if (++idx[d] < self->shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_RETURN_FALSE;
}

static PyObject *NdArray_all(NdArrayObject *self, PyObject *Py_UNUSED(ignored)) {
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
        if (get_element(self->data, self->dtype, off) == 0.0) Py_RETURN_FALSE;
        for (int d = self->ndim - 1; d >= 0; d--) {
            if (++idx[d] < self->shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_RETURN_TRUE;
}

static PyObject *NdArray_fill(NdArrayObject *self, PyObject *val) {
    double v;
    if (PyBool_Check(val)) v = (val == Py_True) ? 1.0 : 0.0;
    else if (PyLong_Check(val)) v = (double)PyLong_AsLongLong(val);
    else if (PyFloat_Check(val)) v = PyFloat_AsDouble(val);
    else { PyErr_SetString(PyExc_TypeError, "fill value must be scalar"); return NULL; }
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
        set_element(self->data, self->dtype, off, v);
        for (int d = self->ndim - 1; d >= 0; d--) {
            if (++idx[d] < self->shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_RETURN_NONE;
}

static PyObject *NdArray_sum(NdArrayObject *self, PyObject *args, PyObject *kwargs) {
    int axis, keepdims;
    if (_parse_axis_keepdims(args, kwargs, &axis, &keepdims) < 0) return NULL;
    return (PyObject*)reduce_op(self, axis, keepdims, 0.0, _comb_add, 0);
}

static PyObject *NdArray_mean(NdArrayObject *self, PyObject *args, PyObject *kwargs) {
    int axis, keepdims;
    if (_parse_axis_keepdims(args, kwargs, &axis, &keepdims) < 0) return NULL;
    return (PyObject*)reduce_op(self, axis, keepdims, 0.0, _comb_add, 1);
}

static PyObject *NdArray_var(NdArrayObject *self, PyObject *args, PyObject *kwargs) {
    int axis, keepdims;
    if (_parse_axis_keepdims(args, kwargs, &axis, &keepdims) < 0) return NULL;
    return (PyObject*)var_op(self, axis, keepdims);
}

static PyObject *NdArray_view(NdArrayObject *self, PyObject *args) {
    PyObject *shape_tuple, *strides_tuple;
    Py_ssize_t offset = 0;
    if (!PyArg_ParseTuple(args, "OO|n", &shape_tuple, &strides_tuple, &offset))
        return NULL;
    int ndim = (int)PyTuple_Size(shape_tuple);
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = ndim;
    r->dtype = self->dtype;
    r->data = self->data + offset;
    for (int i = 0; i < ndim; i++) {
        r->shape[i] = PyLong_AsSsize_t(PyTuple_GetItem(shape_tuple, i));
        r->strides[i] = PyLong_AsSsize_t(PyTuple_GetItem(strides_tuple, i));
    }
    r->size = compute_size(r->shape, ndim);
    r->base = self->base ? self->base : (PyObject*)self;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyMethodDef NdArray_methods[] = {
    {"astype",  (PyCFunction)NdArray_astype,  METH_VARARGS|METH_KEYWORDS, NULL},
    {"copy",    (PyCFunction)NdArray_copy,    METH_NOARGS,  NULL},
    {"reshape", (PyCFunction)NdArray_reshape, METH_VARARGS, NULL},
    {"item",    (PyCFunction)NdArray_item,    METH_NOARGS,  NULL},
    {"any",     (PyCFunction)NdArray_any,     METH_NOARGS,  NULL},
    {"all",     (PyCFunction)NdArray_all,     METH_NOARGS,  NULL},
    {"fill",    (PyCFunction)NdArray_fill,    METH_O,       NULL},
    {"sum",     (PyCFunction)NdArray_sum,     METH_VARARGS|METH_KEYWORDS, NULL},
    {"mean",    (PyCFunction)NdArray_mean,    METH_VARARGS|METH_KEYWORDS, NULL},
    {"var",     (PyCFunction)NdArray_var,     METH_VARARGS|METH_KEYWORDS, NULL},
    {"_view",   (PyCFunction)NdArray_view,    METH_VARARGS, NULL},
    {"__format__", (PyCFunction)NdArray_format, METH_VARARGS, NULL},
    {NULL}
};

/* ── Section 13: NdArray getset properties ───────────────────────────── */

static PyObject *NdArray_get_shape(NdArrayObject *self, void *c) {
    (void)c;
    PyObject *t = PyTuple_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyTuple_SET_ITEM(t, i, PyLong_FromSsize_t(self->shape[i]));
    return t;
}

static PyObject *NdArray_get_dtype(NdArrayObject *self, void *c) {
    (void)c;
    Py_INCREF(dtype_singletons[self->dtype]);
    return (PyObject*)dtype_singletons[self->dtype];
}

static PyObject *NdArray_get_ndim(NdArrayObject *self, void *c) {
    (void)c; return PyLong_FromLong(self->ndim);
}

static PyObject *NdArray_get_T(NdArrayObject *self, void *c) {
    (void)c;
    if (self->ndim <= 1) { Py_INCREF(self); return (PyObject*)self; }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = self->ndim;
    r->dtype = self->dtype;
    r->data = self->data;
    r->size = self->size;
    for (int i = 0; i < self->ndim; i++) {
        r->shape[i] = self->shape[self->ndim - 1 - i];
        r->strides[i] = self->strides[self->ndim - 1 - i];
    }
    r->base = self->base ? self->base : (PyObject*)self;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyObject *NdArray_get_itemsize(NdArrayObject *self, void *c) {
    (void)c; return PyLong_FromLong(dtype_sizes[self->dtype]);
}

static PyObject *NdArray_get_strides_prop(NdArrayObject *self, void *c) {
    (void)c;
    PyObject *t = PyTuple_New(self->ndim);
    for (int i = 0; i < self->ndim; i++)
        PyTuple_SET_ITEM(t, i, PyLong_FromSsize_t(self->strides[i]));
    return t;
}

static PyObject *NdArray_get_size(NdArrayObject *self, void *c) {
    (void)c; return PyLong_FromSsize_t(self->size);
}

static PyObject *NdArray_get_flags(NdArrayObject *self, void *c) {
    (void)c;
    PyObject *d = PyDict_New();
    PyDict_SetItemString(d, "C_CONTIGUOUS", is_c_contiguous(self) ? Py_True : Py_False);
    return d;
}

static PyObject *NdArray_get_flat(NdArrayObject *self, void *c) {
    (void)c;
    NdArrayObject *src = self;
    int need_copy = 0;
    if (!is_c_contiguous(self)) {
        src = make_contiguous_copy(self);
        if (!src) return NULL;
        need_copy = 1;
    }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) { if (need_copy) Py_DECREF(src); return NULL; }
    r->ndim = 1;
    r->dtype = src->dtype;
    r->data = src->data;
    r->shape[0] = src->size;
    r->size = src->size;
    r->strides[0] = dtype_sizes[src->dtype];
    if (src->base) { r->base = src->base; Py_INCREF(r->base); }
    else { r->base = (PyObject*)src; Py_INCREF(src); }
    if (need_copy) Py_DECREF(src);
    return (PyObject*)r;
}

static PyGetSetDef NdArray_getset[] = {
    {"shape",    (getter)NdArray_get_shape,        NULL, NULL, NULL},
    {"dtype",    (getter)NdArray_get_dtype,         NULL, NULL, NULL},
    {"ndim",     (getter)NdArray_get_ndim,          NULL, NULL, NULL},
    {"T",        (getter)NdArray_get_T,             NULL, NULL, NULL},
    {"itemsize", (getter)NdArray_get_itemsize,      NULL, NULL, NULL},
    {"strides",  (getter)NdArray_get_strides_prop,  NULL, NULL, NULL},
    {"size",     (getter)NdArray_get_size,           NULL, NULL, NULL},
    {"flags",    (getter)NdArray_get_flags,          NULL, NULL, NULL},
    {"flat",     (getter)NdArray_get_flat,           NULL, NULL, NULL},
    {NULL}
};

/* ── Section 14: NdArray repr ────────────────────────────────────────── */

static void _format_val(char *buf, size_t bufsz, int dtype, double v) {
    if (dtype == DTYPE_BOOL) {
        snprintf(buf, bufsz, "%s", v != 0.0 ? " True" : "False");
    } else if (dtype >= DTYPE_FLOAT16) {
        snprintf(buf, bufsz, "%g", v);
        /* Ensure a decimal point for float types */
        if (!strchr(buf, '.') && !strchr(buf, 'e') && !strchr(buf, 'n') && !strchr(buf, 'i'))
            strncat(buf, ".", bufsz - strlen(buf) - 1);
    } else {
        snprintf(buf, bufsz, "%lld", (long long)(int64_t)v);
    }
}

static int _repr_recursive(NdArrayObject *self, char *buf, size_t bufsz,
                           size_t *pos, Py_ssize_t *idx, int dim) {
    if (dim == self->ndim) {
        Py_ssize_t off = 0;
        for (int d = 0; d < self->ndim; d++) off += idx[d]*self->strides[d];
        char tmp[64];
        _format_val(tmp, sizeof(tmp), self->dtype, get_element(self->data, self->dtype, off));
        size_t len = strlen(tmp);
        if (*pos + len >= bufsz - 1) return -1;
        memcpy(buf + *pos, tmp, len);
        *pos += len;
        return 0;
    }
    if (*pos >= bufsz - 2) return -1;
    buf[(*pos)++] = '[';
    for (Py_ssize_t i = 0; i < self->shape[dim]; i++) {
        if (i > 0) {
            if (*pos >= bufsz - 3) return -1;
            buf[(*pos)++] = ',';
            buf[(*pos)++] = ' ';
        }
        idx[dim] = i;
        if (_repr_recursive(self, buf, bufsz, pos, idx, dim + 1) < 0) return -1;
    }
    if (*pos >= bufsz - 2) return -1;
    buf[(*pos)++] = ']';
    return 0;
}

static PyObject *NdArray_repr(NdArrayObject *self) {
    if (self->ndim == 0) {
        char tmp[64];
        _format_val(tmp, sizeof(tmp), self->dtype, get_element(self->data, self->dtype, 0));
        return PyUnicode_FromFormat("array(%s)", tmp);
    }
    size_t bufsz = (size_t)self->size * 32 + (size_t)self->ndim * 4 + 64;
    if (bufsz > 1024 * 1024) bufsz = 1024 * 1024;
    char *buf = (char*)malloc(bufsz);
    if (!buf) { PyErr_NoMemory(); return NULL; }
    size_t pos = 0;
    memcpy(buf, "array(", 6); pos = 6;
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    _repr_recursive(self, buf, bufsz, &pos, idx, 0);
    if (pos < bufsz - 2) { buf[pos++] = ')'; }
    buf[pos] = '\0';
    PyObject *r = PyUnicode_FromString(buf);
    free(buf);
    return r;
}

static PyObject *NdArray_format(NdArrayObject *self, PyObject *args) {
    const char *spec = "";
    if (!PyArg_ParseTuple(args, "|s", &spec)) return NULL;
    if (spec[0] == '\0') return NdArray_repr(self);
    if (self->ndim == 0) {
        double v = get_element(self->data, self->dtype, 0);
        PyObject *pyv = PyFloat_FromDouble(v);
        PyObject *fmt = PyUnicode_FromFormat("{:%s}", spec);
        if (!pyv || !fmt) { Py_XDECREF(pyv); Py_XDECREF(fmt); return NULL; }
        PyObject *r = PyObject_CallMethod(pyv, "__format__", "s", spec);
        Py_DECREF(pyv); Py_DECREF(fmt);
        return r;
    }
    return NdArray_repr(self);
}

/* ── Section 15: NdArray number protocol ─────────────────────────────── */

static PyObject *_nb_binop(PyObject *a, PyObject *b, binary_fn fn) {
    NdArrayObject *na = to_ndarray(a);
    NdArrayObject *nb = to_ndarray(b);
    if (!na || !nb) { Py_XDECREF(na); Py_XDECREF(nb); Py_RETURN_NOTIMPLEMENTED; }
    int dt = promote_dtype(na->dtype, nb->dtype);
    NdArrayObject *r = binary_op(na, nb, fn, dt);
    Py_DECREF(na); Py_DECREF(nb);
    return (PyObject*)r;
}

static PyObject *NdArray_add(PyObject *a, PyObject *b) { return _nb_binop(a, b, _op_add); }
static PyObject *NdArray_sub(PyObject *a, PyObject *b) { return _nb_binop(a, b, _op_sub); }
static PyObject *NdArray_mul(PyObject *a, PyObject *b) { return _nb_binop(a, b, _op_mul); }
static PyObject *NdArray_div(PyObject *a, PyObject *b) { return _nb_binop(a, b, _op_div); }

static PyObject *NdArray_neg(PyObject *a) {
    NdArrayObject *na = (NdArrayObject*)a;
    return (PyObject*)unary_op(na, _op_neg, na->dtype);
}

static PyObject *NdArray_abs_nb(PyObject *a) {
    NdArrayObject *na = (NdArrayObject*)a;
    return (PyObject*)unary_op(na, _op_abs, na->dtype);
}

static PyObject *NdArray_matmul(PyObject *a, PyObject *b) {
    if (!PyObject_TypeCheck(a, &NdArrayType) || !PyObject_TypeCheck(b, &NdArrayType))
        Py_RETURN_NOTIMPLEMENTED;
    NdArrayObject *la = (NdArrayObject*)a, *lb = (NdArrayObject*)b;
    if (la->ndim != 2 || lb->ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "matmul: only 2D arrays supported");
        return NULL;
    }
    if (la->shape[1] != lb->shape[0]) {
        PyErr_SetString(PyExc_ValueError, "matmul: shape mismatch");
        return NULL;
    }
    Py_ssize_t M = la->shape[0], K = la->shape[1], N = lb->shape[1];
    int dt = promote_dtype(la->dtype, lb->dtype);
    Py_ssize_t out_shape[2] = {M, N};
    NdArrayObject *r = ndarray_new_empty(dt, out_shape, 2);
    if (!r) return NULL;
    int oes = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < M; i++) {
        for (Py_ssize_t j = 0; j < N; j++) {
            double s = 0.0;
            for (Py_ssize_t k = 0; k < K; k++) {
                double va = get_element(la->data, la->dtype,
                    i * la->strides[0] + k * la->strides[1]);
                double vb = get_element(lb->data, lb->dtype,
                    k * lb->strides[0] + j * lb->strides[1]);
                s += va * vb;
            }
            set_element(r->data, dt, (i * N + j) * oes, s);
        }
    }
    return (PyObject*)r;
}

static int NdArray_bool_nb(PyObject *self) {
    NdArrayObject *a = (NdArrayObject*)self;
    if (a->size != 1) {
        PyErr_SetString(PyExc_ValueError,
            "The truth value of an array with more than one element is ambiguous");
        return -1;
    }
    return get_element(a->data, a->dtype, 0) != 0.0;
}

static PyNumberMethods NdArray_as_number = {
    .nb_add          = NdArray_add,
    .nb_subtract     = NdArray_sub,
    .nb_multiply     = NdArray_mul,
    .nb_true_divide  = NdArray_div,
    .nb_negative     = NdArray_neg,
    .nb_absolute     = NdArray_abs_nb,
    .nb_matrix_multiply = NdArray_matmul,
    .nb_bool         = NdArray_bool_nb,
};

/* ── Section 16: NdArray richcompare ─────────────────────────────────── */

static PyObject *NdArray_richcompare(PyObject *self, PyObject *other, int op) {
    NdArrayObject *a = to_ndarray(self);
    NdArrayObject *b = to_ndarray(other);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b); Py_RETURN_NOTIMPLEMENTED; }
    binary_fn fn;
    switch (op) {
        case Py_EQ: fn = _op_eq; break;
        case Py_NE: fn = _op_ne; break;
        case Py_LT: fn = _op_lt; break;
        case Py_LE: fn = _op_le; break;
        case Py_GT: fn = _op_gt; break;
        case Py_GE: fn = _op_ge; break;
        default: Py_DECREF(a); Py_DECREF(b); Py_RETURN_NOTIMPLEMENTED;
    }
    NdArrayObject *r = binary_op(a, b, fn, DTYPE_BOOL);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

/* ── Section 17: NdArray mapping protocol (getitem / setitem) ────────── */

static Py_ssize_t NdArray_length(NdArrayObject *self) {
    if (self->ndim == 0) return 0;
    return self->shape[0];
}

/* Helper: assign src_val (scalar double) or src_arr to a view region */
static int _assign_to_view(NdArrayObject *dst, PyObject *value) {
    if (PyFloat_Check(value) || PyLong_Check(value) || PyBool_Check(value)) {
        double v;
        if (PyBool_Check(value)) v = (value == Py_True) ? 1.0 : 0.0;
        else if (PyLong_Check(value)) v = (double)PyLong_AsLongLong(value);
        else v = PyFloat_AsDouble(value);
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < dst->size; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < dst->ndim; d++) off += idx[d]*dst->strides[d];
            set_element(dst->data, dst->dtype, off, v);
            for (int d = dst->ndim - 1; d >= 0; d--) {
                if (++idx[d] < dst->shape[d]) break;
                idx[d] = 0;
            }
        }
        return 0;
    }
    NdArrayObject *src = to_ndarray(value);
    if (!src) { PyErr_SetString(PyExc_TypeError, "cannot assign"); return -1; }
    /* broadcast src to dst shape */
    Py_ssize_t bs[NPL_MAXDIM];
    int bndim;
    if (broadcast_shapes(dst->shape, dst->ndim, src->shape, src->ndim, bs, &bndim) < 0) {
        Py_DECREF(src);
        return -1;
    }
    Py_ssize_t src_strides[NPL_MAXDIM], dst_strides[NPL_MAXDIM];
    broadcast_strides(src->shape, src->strides, src->ndim, bndim, src_strides);
    broadcast_strides(dst->shape, dst->strides, dst->ndim, bndim, dst_strides);
    Py_ssize_t total = compute_size(bs, bndim);
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < total; i++) {
        Py_ssize_t soff = 0, doff = 0;
        for (int d = 0; d < bndim; d++) {
            soff += idx[d]*src_strides[d];
            doff += idx[d]*dst_strides[d];
        }
        set_element(dst->data, dst->dtype, doff,
                    get_element(src->data, src->dtype, soff));
        for (int d = bndim - 1; d >= 0; d--) {
            if (++idx[d] < bs[d]) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(src);
    return 0;
}

static Py_ssize_t _normalize_index(Py_ssize_t idx, Py_ssize_t dim_size) {
    if (idx < 0) idx += dim_size;
    return idx;
}

/* Create a view for a single integer index along axis 0 */
static NdArrayObject *_index_view(NdArrayObject *self, Py_ssize_t i) {
    i = _normalize_index(i, self->shape[0]);
    if (i < 0 || i >= self->shape[0]) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return NULL;
    }
    if (self->ndim == 1) return NULL; /* signal: return scalar */
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = self->ndim - 1;
    r->dtype = self->dtype;
    r->data = self->data + i * self->strides[0];
    for (int d = 1; d < self->ndim; d++) {
        r->shape[d - 1] = self->shape[d];
        r->strides[d - 1] = self->strides[d];
    }
    r->size = compute_size(r->shape, r->ndim);
    r->base = self->base ? self->base : (PyObject*)self;
    Py_INCREF(r->base);
    return r;
}

/* Create a view for a slice along axis 0 */
static NdArrayObject *_slice_view(NdArrayObject *self, PyObject *slice_obj) {
    Py_ssize_t start, stop, step, slen;
    if (PySlice_GetIndicesEx(slice_obj, self->shape[0], &start, &stop, &step, &slen) < 0)
        return NULL;
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = self->ndim;
    r->dtype = self->dtype;
    r->data = self->data + start * self->strides[0];
    r->shape[0] = slen;
    r->strides[0] = self->strides[0] * step;
    for (int d = 1; d < self->ndim; d++) {
        r->shape[d] = self->shape[d];
        r->strides[d] = self->strides[d];
    }
    r->size = compute_size(r->shape, r->ndim);
    r->base = self->base ? self->base : (PyObject*)self;
    Py_INCREF(r->base);
    return r;
}

static PyObject *NdArray_getitem(NdArrayObject *self, PyObject *key) {
    /* Integer index */
    if (PyLong_Check(key)) {
        if (self->ndim == 0) {
            PyErr_SetString(PyExc_IndexError, "too many indices for 0-d array");
            return NULL;
        }
        Py_ssize_t i = _normalize_index(PyLong_AsSsize_t(key), self->shape[0]);
        if (i < 0 || i >= self->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return NULL;
        }
        if (self->ndim == 1) {
            double v = get_element(self->data, self->dtype, i * self->strides[0]);
            if (self->dtype == DTYPE_BOOL) return PyBool_FromLong((long)v);
            if (self->dtype >= DTYPE_FLOAT16) return PyFloat_FromDouble(v);
            return PyLong_FromLongLong((long long)(int64_t)v);
        }
        NdArrayObject *r = _index_view(self, i);
        return (PyObject*)r;
    }

    /* Slice */
    if (PySlice_Check(key)) {
        if (self->ndim == 0) {
            PyErr_SetString(PyExc_IndexError, "too many indices for 0-d array");
            return NULL;
        }
        return (PyObject*)_slice_view(self, key);
    }

    /* Bool ndarray mask */
    if (PyObject_TypeCheck(key, &NdArrayType)) {
        NdArrayObject *mask = (NdArrayObject*)key;
        if (mask->dtype == DTYPE_BOOL) {
            /* Count true elements */
            Py_ssize_t count = 0;
            Py_ssize_t midx[NPL_MAXDIM];
            memset(midx, 0, sizeof(midx));
            for (Py_ssize_t i = 0; i < mask->size; i++) {
                Py_ssize_t off = 0;
                for (int d = 0; d < mask->ndim; d++) off += midx[d]*mask->strides[d];
                if (get_element(mask->data, DTYPE_BOOL, off) != 0.0) count++;
                for (int d = mask->ndim - 1; d >= 0; d--) {
                    if (++midx[d] < mask->shape[d]) break;
                    midx[d] = 0;
                }
            }
            NdArrayObject *r = ndarray_new_empty(self->dtype, &count, 1);
            if (!r) return NULL;
            int es = dtype_sizes[self->dtype];
            Py_ssize_t ri = 0;
            Py_ssize_t sidx[NPL_MAXDIM];
            memset(sidx, 0, sizeof(sidx));
            memset(midx, 0, sizeof(midx));
            for (Py_ssize_t i = 0; i < self->size && i < mask->size; i++) {
                Py_ssize_t soff = 0, moff = 0;
                for (int d = 0; d < self->ndim; d++) soff += sidx[d]*self->strides[d];
                for (int d = 0; d < mask->ndim; d++) moff += midx[d]*mask->strides[d];
                if (get_element(mask->data, DTYPE_BOOL, moff) != 0.0) {
                    set_element(r->data, r->dtype, ri * es,
                                get_element(self->data, self->dtype, soff));
                    ri++;
                }
                for (int d = self->ndim - 1; d >= 0; d--) {
                    if (++sidx[d] < self->shape[d]) break;
                    sidx[d] = 0;
                }
                for (int d = mask->ndim - 1; d >= 0; d--) {
                    if (++midx[d] < mask->shape[d]) break;
                    midx[d] = 0;
                }
            }
            return (PyObject*)r;
        }
        /* Integer ndarray: fancy indexing along axis 0 */
        NdArrayObject *idx_arr = mask;
        if (self->ndim == 0) {
            PyErr_SetString(PyExc_IndexError, "too many indices for 0-d array");
            return NULL;
        }
        /* Output shape = idx_arr.shape + self.shape[1:] */
        Py_ssize_t out_shape[NPL_MAXDIM];
        int out_ndim = 0;
        for (int d = 0; d < idx_arr->ndim; d++)
            out_shape[out_ndim++] = idx_arr->shape[d];
        for (int d = 1; d < self->ndim; d++)
            out_shape[out_ndim++] = self->shape[d];
        NdArrayObject *r = ndarray_new_empty(self->dtype, out_shape, out_ndim);
        if (!r) return NULL;
        /* Size of each sub-array along axis 0 */
        Py_ssize_t sub_size = 1;
        for (int d = 1; d < self->ndim; d++) sub_size *= self->shape[d];
        int es = dtype_sizes[self->dtype];
        Py_ssize_t iidx[NPL_MAXDIM];
        memset(iidx, 0, sizeof(iidx));
        Py_ssize_t out_flat = 0;
        for (Py_ssize_t ii = 0; ii < idx_arr->size; ii++) {
            Py_ssize_t ioff = 0;
            for (int d = 0; d < idx_arr->ndim; d++) ioff += iidx[d]*idx_arr->strides[d];
            Py_ssize_t idx_val = (Py_ssize_t)(int64_t)get_element(idx_arr->data, idx_arr->dtype, ioff);
            idx_val = _normalize_index(idx_val, self->shape[0]);
            /* Copy sub-array at self[idx_val] */
            Py_ssize_t sub_idx[NPL_MAXDIM];
            memset(sub_idx, 0, sizeof(sub_idx));
            for (Py_ssize_t si = 0; si < sub_size; si++) {
                Py_ssize_t src_off = idx_val * self->strides[0];
                for (int d = 1; d < self->ndim; d++) src_off += sub_idx[d-1]*self->strides[d];
                set_element(r->data, r->dtype, out_flat * es,
                            get_element(self->data, self->dtype, src_off));
                out_flat++;
                for (int d = self->ndim - 2; d >= 0; d--) {
                    if (++sub_idx[d] < self->shape[d + 1]) break;
                    sub_idx[d] = 0;
                }
            }
            for (int d = idx_arr->ndim - 1; d >= 0; d--) {
                if (++iidx[d] < idx_arr->shape[d]) break;
                iidx[d] = 0;
            }
        }
        return (PyObject*)r;
    }

    /* Tuple of indices */
    if (PyTuple_Check(key)) {
        Py_ssize_t nkeys = PyTuple_Size(key);
        if (nkeys == 0) { Py_INCREF(self); return (PyObject*)self; }

        /* Build the view step by step */
        NdArrayObject *cur = self;
        Py_INCREF(cur);
        for (Py_ssize_t ki = 0; ki < nkeys; ki++) {
            PyObject *k = PyTuple_GetItem(key, ki);
            if (PyLong_Check(k)) {
                Py_ssize_t idx = _normalize_index(PyLong_AsSsize_t(k), cur->shape[0]);
                if (idx < 0 || idx >= cur->shape[0]) {
                    Py_DECREF(cur);
                    PyErr_SetString(PyExc_IndexError, "index out of bounds");
                    return NULL;
                }
                if (cur->ndim == 1) {
                    double v = get_element(cur->data, cur->dtype, idx * cur->strides[0]);
                    Py_DECREF(cur);
                    if (cur->dtype == DTYPE_BOOL) return PyBool_FromLong((long)v);
                    if (cur->dtype >= DTYPE_FLOAT16) return PyFloat_FromDouble(v);
                    return PyLong_FromLongLong((long long)(int64_t)v);
                }
                NdArrayObject *next = _index_view(cur, idx);
                Py_DECREF(cur);
                if (!next) return NULL;
                cur = next;
            } else if (PySlice_Check(k)) {
                NdArrayObject *next = _slice_view(cur, k);
                Py_DECREF(cur);
                if (!next) return NULL;
                cur = next;
            } else {
                Py_DECREF(cur);
                PyErr_SetString(PyExc_IndexError, "unsupported index type in tuple");
                return NULL;
            }
        }
        return (PyObject*)cur;
    }

    PyErr_SetString(PyExc_IndexError, "unsupported index type");
    return NULL;
}

static int NdArray_setitem(NdArrayObject *self, PyObject *key, PyObject *value) {
    /* Integer index */
    if (PyLong_Check(key)) {
        if (self->ndim == 0) {
            PyErr_SetString(PyExc_IndexError, "too many indices for 0-d array");
            return -1;
        }
        Py_ssize_t i = _normalize_index(PyLong_AsSsize_t(key), self->shape[0]);
        if (i < 0 || i >= self->shape[0]) {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return -1;
        }
        if (self->ndim == 1) {
            double v;
            if (PyBool_Check(value)) v = (value == Py_True) ? 1.0 : 0.0;
            else if (PyLong_Check(value)) v = (double)PyLong_AsLongLong(value);
            else if (PyFloat_Check(value)) v = PyFloat_AsDouble(value);
            else { PyErr_SetString(PyExc_TypeError, "cannot assign"); return -1; }
            set_element(self->data, self->dtype, i * self->strides[0], v);
            return 0;
        }
        NdArrayObject *view = _index_view(self, i);
        if (!view) return -1;
        int r = _assign_to_view(view, value);
        Py_DECREF(view);
        return r;
    }

    /* Slice */
    if (PySlice_Check(key)) {
        NdArrayObject *view = _slice_view(self, key);
        if (!view) return -1;
        int r = _assign_to_view(view, value);
        Py_DECREF(view);
        return r;
    }

    /* Bool ndarray mask */
    if (PyObject_TypeCheck(key, &NdArrayType)) {
        NdArrayObject *mask = (NdArrayObject*)key;
        if (mask->dtype == DTYPE_BOOL) {
            double v = 0;
            int is_scalar = 0;
            NdArrayObject *val_arr = NULL;
            if (PyBool_Check(value)) { v = (value == Py_True) ? 1.0 : 0.0; is_scalar = 1; }
            else if (PyLong_Check(value)) { v = (double)PyLong_AsLongLong(value); is_scalar = 1; }
            else if (PyFloat_Check(value)) { v = PyFloat_AsDouble(value); is_scalar = 1; }
            else {
                val_arr = to_ndarray(value);
                if (!val_arr) { PyErr_SetString(PyExc_TypeError, "bad value"); return -1; }
            }
            Py_ssize_t sidx[NPL_MAXDIM], midx[NPL_MAXDIM];
            memset(sidx, 0, sizeof(sidx));
            memset(midx, 0, sizeof(midx));
            Py_ssize_t vi = 0;
            int ves = val_arr ? dtype_sizes[val_arr->dtype] : 0;
            for (Py_ssize_t i = 0; i < self->size && i < mask->size; i++) {
                Py_ssize_t soff = 0, moff = 0;
                for (int d = 0; d < self->ndim; d++) soff += sidx[d]*self->strides[d];
                for (int d = 0; d < mask->ndim; d++) moff += midx[d]*mask->strides[d];
                if (get_element(mask->data, DTYPE_BOOL, moff) != 0.0) {
                    double sv = is_scalar ? v : get_element(val_arr->data, val_arr->dtype, vi * ves);
                    set_element(self->data, self->dtype, soff, sv);
                    vi++;
                }
                for (int d = self->ndim - 1; d >= 0; d--) {
                    if (++sidx[d] < self->shape[d]) break;
                    sidx[d] = 0;
                }
                for (int d = mask->ndim - 1; d >= 0; d--) {
                    if (++midx[d] < mask->shape[d]) break;
                    midx[d] = 0;
                }
            }
            Py_XDECREF(val_arr);
            return 0;
        }
        /* Integer ndarray */
        NdArrayObject *idx_arr = mask;
        NdArrayObject *val_src = to_ndarray(value);
        int is_scalar = 0;
        double scalar_v = 0;
        if (!val_src) {
            if (PyBool_Check(value)) { scalar_v = (value == Py_True) ? 1.0 : 0.0; is_scalar = 1; }
            else if (PyLong_Check(value)) { scalar_v = (double)PyLong_AsLongLong(value); is_scalar = 1; }
            else if (PyFloat_Check(value)) { scalar_v = PyFloat_AsDouble(value); is_scalar = 1; }
            else { PyErr_SetString(PyExc_TypeError, "bad value"); return -1; }
        }
        Py_ssize_t sub_size = 1;
        for (int d = 1; d < self->ndim; d++) sub_size *= self->shape[d];
        Py_ssize_t iidx[NPL_MAXDIM];
        memset(iidx, 0, sizeof(iidx));
        Py_ssize_t val_flat = 0;
        int val_es = val_src ? dtype_sizes[val_src->dtype] : 0;
        for (Py_ssize_t ii = 0; ii < idx_arr->size; ii++) {
            Py_ssize_t ioff = 0;
            for (int d = 0; d < idx_arr->ndim; d++) ioff += iidx[d]*idx_arr->strides[d];
            Py_ssize_t idx_val = (Py_ssize_t)(int64_t)get_element(idx_arr->data, idx_arr->dtype, ioff);
            idx_val = _normalize_index(idx_val, self->shape[0]);
            Py_ssize_t sub_idx[NPL_MAXDIM];
            memset(sub_idx, 0, sizeof(sub_idx));
            for (Py_ssize_t si = 0; si < sub_size; si++) {
                Py_ssize_t dst_off = idx_val * self->strides[0];
                for (int d = 1; d < self->ndim; d++) dst_off += sub_idx[d-1]*self->strides[d];
                double sv;
                if (is_scalar) sv = scalar_v;
                else sv = get_element(val_src->data, val_src->dtype, val_flat * val_es);
                set_element(self->data, self->dtype, dst_off, sv);
                val_flat++;
                for (int d = self->ndim - 2; d >= 0; d--) {
                    if (++sub_idx[d] < self->shape[d + 1]) break;
                    sub_idx[d] = 0;
                }
            }
            for (int d = idx_arr->ndim - 1; d >= 0; d--) {
                if (++iidx[d] < idx_arr->shape[d]) break;
                iidx[d] = 0;
            }
        }
        Py_XDECREF(val_src);
        return 0;
    }

    /* Tuple */
    if (PyTuple_Check(key)) {
        Py_ssize_t nkeys = PyTuple_Size(key);
        if (nkeys == 0) return _assign_to_view(self, value);
        NdArrayObject *cur = self;
        Py_INCREF(cur);
        for (Py_ssize_t ki = 0; ki < nkeys; ki++) {
            PyObject *k = PyTuple_GetItem(key, ki);
            if (PyLong_Check(k)) {
                Py_ssize_t idx = _normalize_index(PyLong_AsSsize_t(k), cur->shape[0]);
                if (idx < 0 || idx >= cur->shape[0]) {
                    Py_DECREF(cur);
                    PyErr_SetString(PyExc_IndexError, "index out of bounds");
                    return -1;
                }
                if (cur->ndim == 1) {
                    double v;
                    if (PyBool_Check(value)) v = (value == Py_True) ? 1.0 : 0.0;
                    else if (PyLong_Check(value)) v = (double)PyLong_AsLongLong(value);
                    else if (PyFloat_Check(value)) v = PyFloat_AsDouble(value);
                    else { Py_DECREF(cur); PyErr_SetString(PyExc_TypeError, "cannot assign"); return -1; }
                    set_element(cur->data, cur->dtype, idx * cur->strides[0], v);
                    Py_DECREF(cur);
                    return 0;
                }
                NdArrayObject *next = _index_view(cur, idx);
                Py_DECREF(cur);
                if (!next) return -1;
                cur = next;
            } else if (PySlice_Check(k)) {
                NdArrayObject *next = _slice_view(cur, k);
                Py_DECREF(cur);
                if (!next) return -1;
                cur = next;
            } else {
                Py_DECREF(cur);
                PyErr_SetString(PyExc_IndexError, "unsupported index type");
                return -1;
            }
        }
        int r = _assign_to_view(cur, value);
        Py_DECREF(cur);
        return r;
    }

    PyErr_SetString(PyExc_IndexError, "unsupported index type");
    return -1;
}

static PyMappingMethods NdArray_as_mapping = {
    .mp_length       = (lenfunc)NdArray_length,
    .mp_subscript     = (binaryfunc)NdArray_getitem,
    .mp_ass_subscript = (objobjargproc)NdArray_setitem,
};

/* ── Section 18: NdArray type object ─────────────────────────────────── */

static PyTypeObject NdArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name        = "numpy.ndarray",
    .tp_basicsize   = sizeof(NdArrayObject),
    .tp_dealloc     = (destructor)NdArray_dealloc,
    .tp_repr        = (reprfunc)NdArray_repr,
    .tp_as_number   = &NdArray_as_number,
    .tp_as_mapping  = &NdArray_as_mapping,
    .tp_flags       = Py_TPFLAGS_DEFAULT,
    .tp_methods     = NdArray_methods,
    .tp_getset      = NdArray_getset,
    .tp_richcompare = NdArray_richcompare,
};

/* ── Section 19: Module-level functions ──────────────────────────────── */

/* Helper to parse shape from a Python object (int, tuple, list) */
static int _parse_shape_arg(PyObject *arg, Py_ssize_t *shape, int *ndim) {
    if (PyLong_Check(arg)) {
        *ndim = 1;
        shape[0] = PyLong_AsSsize_t(arg);
        return 0;
    }
    if (PyTuple_Check(arg) || PyList_Check(arg)) {
        Py_ssize_t n = PySequence_Size(arg);
        *ndim = (int)n;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PySequence_GetItem(arg, i);
            shape[i] = PyLong_AsSsize_t(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) return -1;
        }
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "shape must be int or tuple");
    return -1;
}

static PyObject *mod_array(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"object", "dtype", "copy", NULL};
    PyObject *obj, *dt_obj = Py_None, *copy_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &obj, &dt_obj, &copy_obj))
        return NULL;
    int force_dt = -1;
    if (dt_obj != Py_None) {
        force_dt = _parse_dtype_enum(dt_obj);
        if (force_dt < 0) return NULL;
    }
    /* ndarray input */
    if (PyObject_TypeCheck(obj, &NdArrayType)) {
        NdArrayObject *src = (NdArrayObject*)obj;
        int dt = (force_dt >= 0) ? force_dt : src->dtype;
        NdArrayObject *r = ndarray_new_empty(dt, src->shape, src->ndim);
        if (!r) return NULL;
        int oes = dtype_sizes[dt];
        Py_ssize_t idx[NPL_MAXDIM];
        memset(idx, 0, sizeof(idx));
        for (Py_ssize_t i = 0; i < src->size; i++) {
            Py_ssize_t off = 0;
            for (int d = 0; d < src->ndim; d++) off += idx[d]*src->strides[d];
            set_element(r->data, dt, i * oes, get_element(src->data, src->dtype, off));
            for (int d = src->ndim - 1; d >= 0; d--) {
                if (++idx[d] < src->shape[d]) break;
                idx[d] = 0;
            }
        }
        return (PyObject*)r;
    }
    /* Scalar */
    if (PyBool_Check(obj) || PyLong_Check(obj) || PyFloat_Check(obj)) {
        int dt;
        double v;
        if (PyBool_Check(obj)) { dt = DTYPE_BOOL; v = (obj == Py_True) ? 1.0 : 0.0; }
        else if (PyLong_Check(obj)) { dt = DTYPE_INT64; v = (double)PyLong_AsLongLong(obj); }
        else { dt = DTYPE_FLOAT64; v = PyFloat_AsDouble(obj); }
        if (force_dt >= 0) dt = force_dt;
        NdArrayObject *r = ndarray_new_empty(dt, NULL, 0);
        if (!r) return NULL;
        set_element(r->data, dt, 0, v);
        return (PyObject*)r;
    }
    /* List/tuple */
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        Py_ssize_t shape[NPL_MAXDIM];
        int ndim;
        if (_infer_shape(obj, shape, &ndim) < 0) return NULL;
        int dt = (force_dt >= 0) ? force_dt : _infer_dtype(obj, 0, ndim);
        if (dt < 0) return NULL;
        NdArrayObject *r = ndarray_new_empty(dt, shape, ndim);
        if (!r) return NULL;
        Py_ssize_t fi = 0;
        if (_fill_from_seq(r, obj, 0, &fi) < 0) { Py_DECREF(r); return NULL; }
        return (PyObject*)r;
    }
    PyErr_SetString(PyExc_TypeError, "cannot create array from object");
    return NULL;
}

static PyObject *mod_asarray(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "dtype", "copy", NULL};
    PyObject *obj, *dt_obj = Py_None, *copy_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &obj, &dt_obj, &copy_obj))
        return NULL;
    int force_dt = -1;
    if (dt_obj != Py_None) {
        force_dt = _parse_dtype_enum(dt_obj);
        if (force_dt < 0) return NULL;
    }
    if (PyObject_TypeCheck(obj, &NdArrayType)) {
        NdArrayObject *src = (NdArrayObject*)obj;
        if (force_dt < 0 || force_dt == src->dtype) {
            Py_INCREF(obj);
            return obj;
        }
    }
    return mod_array(self, args, kwargs);
}

static PyObject *mod_zeros(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"shape", "dtype", NULL};
    PyObject *shape_obj, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &shape_obj, &dt_obj))
        return NULL;
    Py_ssize_t shape[NPL_MAXDIM]; int ndim;
    if (_parse_shape_arg(shape_obj, shape, &ndim) < 0) return NULL;
    int dt = DTYPE_FLOAT64;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    return (PyObject*)ndarray_new_empty(dt, shape, ndim);
}

static PyObject *mod_ones(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"shape", "dtype", NULL};
    PyObject *shape_obj, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &shape_obj, &dt_obj))
        return NULL;
    Py_ssize_t shape[NPL_MAXDIM]; int ndim;
    if (_parse_shape_arg(shape_obj, shape, &ndim) < 0) return NULL;
    int dt = DTYPE_FLOAT64;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    NdArrayObject *r = ndarray_new_empty(dt, shape, ndim);
    if (!r) return NULL;
    int es = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < r->size; i++)
        set_element(r->data, dt, i * es, 1.0);
    return (PyObject*)r;
}

static PyObject *mod_full(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"shape", "fill_value", "dtype", NULL};
    PyObject *shape_obj, *fv_obj, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist, &shape_obj, &fv_obj, &dt_obj))
        return NULL;
    Py_ssize_t shape[NPL_MAXDIM]; int ndim;
    if (_parse_shape_arg(shape_obj, shape, &ndim) < 0) return NULL;
    double fv;
    if (PyBool_Check(fv_obj)) fv = (fv_obj == Py_True) ? 1.0 : 0.0;
    else if (PyLong_Check(fv_obj)) fv = (double)PyLong_AsLongLong(fv_obj);
    else if (PyFloat_Check(fv_obj)) fv = PyFloat_AsDouble(fv_obj);
    else { PyErr_SetString(PyExc_TypeError, "fill_value must be scalar"); return NULL; }
    int dt = DTYPE_FLOAT64;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    NdArrayObject *r = ndarray_new_empty(dt, shape, ndim);
    if (!r) return NULL;
    int es = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < r->size; i++)
        set_element(r->data, dt, i * es, fv);
    return (PyObject*)r;
}

static PyObject *mod_empty(PyObject *self, PyObject *args, PyObject *kwargs) {
    return mod_zeros(self, args, kwargs);
}

static PyObject *mod_empty_like(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "dtype", NULL};
    PyObject *obj, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &dt_obj))
        return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *src = (NdArrayObject*)obj;
    int dt = src->dtype;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    return (PyObject*)ndarray_new_empty(dt, src->shape, src->ndim);
}

static PyObject *mod_zeros_like(PyObject *self, PyObject *args, PyObject *kwargs) {
    return mod_empty_like(self, args, kwargs);
}

static PyObject *mod_ones_like(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "dtype", NULL};
    PyObject *obj, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &dt_obj))
        return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *src = (NdArrayObject*)obj;
    int dt = src->dtype;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    NdArrayObject *r = ndarray_new_empty(dt, src->shape, src->ndim);
    if (!r) return NULL;
    int es = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < r->size; i++)
        set_element(r->data, dt, i * es, 1.0);
    return (PyObject*)r;
}

static PyObject *mod_arange(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"start", "stop", "step", "dtype", NULL};
    PyObject *a1 = NULL, *a2 = NULL, *a3 = NULL, *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", kwlist, &a1, &a2, &a3, &dt_obj))
        return NULL;
    double start = 0, stop, step = 1;
    int any_float = 0;
    if (a2 == NULL) {
        /* arange(stop) */
        if (PyFloat_Check(a1)) { stop = PyFloat_AsDouble(a1); any_float = 1; }
        else stop = (double)PyLong_AsLongLong(a1);
    } else if (a3 == NULL) {
        /* arange(start, stop) */
        if (PyFloat_Check(a1)) { start = PyFloat_AsDouble(a1); any_float = 1; }
        else start = (double)PyLong_AsLongLong(a1);
        if (PyFloat_Check(a2)) { stop = PyFloat_AsDouble(a2); any_float = 1; }
        else stop = (double)PyLong_AsLongLong(a2);
    } else {
        if (PyFloat_Check(a1)) { start = PyFloat_AsDouble(a1); any_float = 1; }
        else start = (double)PyLong_AsLongLong(a1);
        if (PyFloat_Check(a2)) { stop = PyFloat_AsDouble(a2); any_float = 1; }
        else stop = (double)PyLong_AsLongLong(a2);
        if (PyFloat_Check(a3)) { step = PyFloat_AsDouble(a3); any_float = 1; }
        else step = (double)PyLong_AsLongLong(a3);
    }
    int dt = any_float ? DTYPE_FLOAT64 : DTYPE_INT64;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    Py_ssize_t n = 0;
    if (step > 0 && stop > start) n = (Py_ssize_t)ceil((stop - start) / step);
    else if (step < 0 && stop < start) n = (Py_ssize_t)ceil((start - stop) / (-step));
    if (n < 0) n = 0;
    NdArrayObject *r = ndarray_new_empty(dt, &n, 1);
    if (!r) return NULL;
    int es = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < n; i++)
        set_element(r->data, dt, i * es, start + i * step);
    return (PyObject*)r;
}

static PyObject *mod_linspace(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"start", "stop", "num", "dtype", NULL};
    double start, stop;
    Py_ssize_t num = 50;
    PyObject *dt_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dd|nO", kwlist, &start, &stop, &num, &dt_obj))
        return NULL;
    int dt = DTYPE_FLOAT64;
    if (dt_obj != Py_None) { dt = _parse_dtype_enum(dt_obj); if (dt < 0) return NULL; }
    NdArrayObject *r = ndarray_new_empty(dt, &num, 1);
    if (!r) return NULL;
    int es = dtype_sizes[dt];
    for (Py_ssize_t i = 0; i < num; i++) {
        double t = (num > 1) ? (double)i / (double)(num - 1) : 0.0;
        set_element(r->data, dt, i * es, start + t * (stop - start));
    }
    return (PyObject*)r;
}

/* Math functions */

static PyObject *mod_exp(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_exp, a->dtype);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_log(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_log, a->dtype);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_sqrt(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_sqrt, a->dtype);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_tanh(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_tanh, a->dtype);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_abs(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_abs, a->dtype);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_clip(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *lo_obj, *hi_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &lo_obj, &hi_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    double lo = -INFINITY, hi = INFINITY;
    if (lo_obj != Py_None) {
        if (PyFloat_Check(lo_obj)) lo = PyFloat_AsDouble(lo_obj);
        else if (PyLong_Check(lo_obj)) lo = (double)PyLong_AsLongLong(lo_obj);
    }
    if (hi_obj != Py_None) {
        if (PyFloat_Check(hi_obj)) hi = PyFloat_AsDouble(hi_obj);
        else if (PyLong_Check(hi_obj)) hi = (double)PyLong_AsLongLong(hi_obj);
    }
    NdArrayObject *r = ndarray_new_empty(a->dtype, a->shape, a->ndim);
    if (!r) { Py_DECREF(a); return NULL; }
    int es = dtype_sizes[a->dtype];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < a->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < a->ndim; d++) off += idx[d]*a->strides[d];
        double v = get_element(a->data, a->dtype, off);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        set_element(r->data, r->dtype, i * es, v);
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_maximum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    int dt = promote_dtype(a->dtype, b->dtype);
    NdArrayObject *r = binary_op(a, b, _op_maximum, dt);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

static PyObject *mod_minimum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    int dt = promote_dtype(a->dtype, b->dtype);
    NdArrayObject *r = binary_op(a, b, _op_minimum, dt);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

static PyObject *mod_where(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *c_obj, *x_obj, *y_obj;
    if (!PyArg_ParseTuple(args, "OOO", &c_obj, &x_obj, &y_obj)) return NULL;
    NdArrayObject *cond = to_ndarray(c_obj);
    NdArrayObject *x = to_ndarray(x_obj);
    NdArrayObject *y = to_ndarray(y_obj);
    if (!cond || !x || !y) {
        Py_XDECREF(cond); Py_XDECREF(x); Py_XDECREF(y);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL;
    }
    /* Broadcast all three */
    Py_ssize_t s1[NPL_MAXDIM], s2[NPL_MAXDIM];
    int n1, n2;
    if (broadcast_shapes(cond->shape, cond->ndim, x->shape, x->ndim, s1, &n1) < 0) {
        Py_DECREF(cond); Py_DECREF(x); Py_DECREF(y); return NULL;
    }
    if (broadcast_shapes(s1, n1, y->shape, y->ndim, s2, &n2) < 0) {
        Py_DECREF(cond); Py_DECREF(x); Py_DECREF(y); return NULL;
    }
    int dt = promote_dtype(x->dtype, y->dtype);
    NdArrayObject *r = ndarray_new_empty(dt, s2, n2);
    if (!r) { Py_DECREF(cond); Py_DECREF(x); Py_DECREF(y); return NULL; }
    Py_ssize_t sc[NPL_MAXDIM], sx[NPL_MAXDIM], sy[NPL_MAXDIM];
    broadcast_strides(cond->shape, cond->strides, cond->ndim, n2, sc);
    broadcast_strides(x->shape, x->strides, x->ndim, n2, sx);
    broadcast_strides(y->shape, y->strides, y->ndim, n2, sy);
    int oes = dtype_sizes[dt];
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    Py_ssize_t total = compute_size(s2, n2);
    for (Py_ssize_t i = 0; i < total; i++) {
        Py_ssize_t oc = 0, ox = 0, oy = 0;
        for (int d = 0; d < n2; d++) { oc += idx[d]*sc[d]; ox += idx[d]*sx[d]; oy += idx[d]*sy[d]; }
        double cv = get_element(cond->data, cond->dtype, oc);
        double val = cv != 0.0 ? get_element(x->data, x->dtype, ox) : get_element(y->data, y->dtype, oy);
        set_element(r->data, dt, i * oes, val);
        for (int d = n2 - 1; d >= 0; d--) {
            if (++idx[d] < s2[d]) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(cond); Py_DECREF(x); Py_DECREF(y);
    return (PyObject*)r;
}

/* Reduction module functions */

static PyObject *mod_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", "keepdims", NULL};
    PyObject *obj, *axis_obj = Py_None;
    int keepdims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &obj, &axis_obj, &keepdims))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_op(a, axis, keepdims, 0.0, _comb_add, 0);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_max(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", "keepdims", NULL};
    PyObject *obj, *axis_obj = Py_None;
    int keepdims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &obj, &axis_obj, &keepdims))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_op(a, axis, keepdims, -INFINITY, _comb_max, 0);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_min(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", "keepdims", NULL};
    PyObject *obj, *axis_obj = Py_None;
    int keepdims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &obj, &axis_obj, &keepdims))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_op(a, axis, keepdims, INFINITY, _comb_min, 0);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_mean(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", "keepdims", NULL};
    PyObject *obj, *axis_obj = Py_None;
    int keepdims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &obj, &axis_obj, &keepdims))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_op(a, axis, keepdims, 0.0, _comb_add, 1);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_var(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", "keepdims", NULL};
    PyObject *obj, *axis_obj = Py_None;
    int keepdims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist, &obj, &axis_obj, &keepdims))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = var_op(a, axis, keepdims);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_argmax(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", NULL};
    PyObject *obj, *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &axis_obj))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_arg_op(a, axis, 1);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_argmin(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", NULL};
    PyObject *obj, *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &axis_obj))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = reduce_arg_op(a, axis, 0);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", NULL};
    PyObject *obj, *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &axis_obj))
        return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    int axis = AXIS_NONE;
    if (axis_obj != Py_None) { axis = (int)PyLong_AsLong(axis_obj); if (PyErr_Occurred()) { Py_DECREF(a); return NULL; } }
    NdArrayObject *r = cumsum_op(a, axis);
    Py_DECREF(a);
    return (PyObject*)r;
}

/* Predicate functions */

static PyObject *mod_isnan(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_isnan_d, DTYPE_BOOL);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_isinf(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_isinf_d, DTYPE_BOOL);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_isposinf(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_isposinf_d, DTYPE_BOOL);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_isneginf(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_isneginf_d, DTYPE_BOOL);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_logical_not(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    NdArrayObject *r = unary_op(a, _op_logical_not_d, DTYPE_BOOL);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_logical_and(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    NdArrayObject *r = binary_op(a, b, _op_logical_and_d, DTYPE_BOOL);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

static PyObject *mod_logical_or(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    NdArrayObject *r = binary_op(a, b, _op_logical_or_d, DTYPE_BOOL);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

static PyObject *mod_logical_xor(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    NdArrayObject *r = binary_op(a, b, _op_logical_xor_d, DTYPE_BOOL);
    Py_DECREF(a); Py_DECREF(b);
    return (PyObject*)r;
}

static PyObject *mod_allclose(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "b", "rtol", "atol", NULL};
    PyObject *a_obj, *b_obj;
    double rtol = 1e-5, atol = 1e-8;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd", kwlist, &a_obj, &b_obj, &rtol, &atol))
        return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    /* Broadcast */
    Py_ssize_t out_shape[NPL_MAXDIM];
    int out_ndim;
    if (broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim) < 0) {
        Py_DECREF(a); Py_DECREF(b); return NULL;
    }
    Py_ssize_t sa[NPL_MAXDIM], sb[NPL_MAXDIM];
    broadcast_strides(a->shape, a->strides, a->ndim, out_ndim, sa);
    broadcast_strides(b->shape, b->strides, b->ndim, out_ndim, sb);
    Py_ssize_t total = compute_size(out_shape, out_ndim);
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < total; i++) {
        Py_ssize_t oa = 0, ob = 0;
        for (int d = 0; d < out_ndim; d++) { oa += idx[d]*sa[d]; ob += idx[d]*sb[d]; }
        double va = get_element(a->data, a->dtype, oa);
        double vb = get_element(b->data, b->dtype, ob);
        double diff = fabs(va - vb);
        if (!(diff <= atol + rtol * fabs(vb))) {
            Py_DECREF(a); Py_DECREF(b); Py_RETURN_FALSE;
        }
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(a); Py_DECREF(b);
    Py_RETURN_TRUE;
}

static PyObject *mod_array_equal(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    NdArrayObject *a = to_ndarray(a_obj), *b = to_ndarray(b_obj);
    if (!a || !b) { Py_XDECREF(a); Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "expected arrays"); return NULL; }
    if (a->ndim != b->ndim) { Py_DECREF(a); Py_DECREF(b); Py_RETURN_FALSE; }
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) { Py_DECREF(a); Py_DECREF(b); Py_RETURN_FALSE; }
    Py_ssize_t aidx[NPL_MAXDIM], bidx[NPL_MAXDIM];
    memset(aidx, 0, sizeof(aidx));
    memset(bidx, 0, sizeof(bidx));
    for (Py_ssize_t i = 0; i < a->size; i++) {
        Py_ssize_t oa = 0, ob = 0;
        for (int d = 0; d < a->ndim; d++) { oa += aidx[d]*a->strides[d]; ob += bidx[d]*b->strides[d]; }
        if (get_element(a->data, a->dtype, oa) != get_element(b->data, b->dtype, ob)) {
            Py_DECREF(a); Py_DECREF(b); Py_RETURN_FALSE;
        }
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (++aidx[d] < a->shape[d]) break;
            aidx[d] = 0;
        }
        for (int d = b->ndim - 1; d >= 0; d--) {
            if (++bidx[d] < b->shape[d]) break;
            bidx[d] = 0;
        }
    }
    Py_DECREF(a); Py_DECREF(b);
    Py_RETURN_TRUE;
}

/* Manipulation functions */

static PyObject *mod_copyto(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *dst_obj, *src_obj;
    if (!PyArg_ParseTuple(args, "OO", &dst_obj, &src_obj)) return NULL;
    if (!PyObject_TypeCheck(dst_obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "dst must be ndarray"); return NULL;
    }
    NdArrayObject *dst = (NdArrayObject*)dst_obj;
    int r = _assign_to_view(dst, src_obj);
    if (r < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *mod_ascontiguousarray(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *a = (NdArrayObject*)obj;
    if (is_c_contiguous(a)) { Py_INCREF(a); return (PyObject*)a; }
    return (PyObject*)make_contiguous_copy(a);
}

static PyObject *mod_broadcast_to(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *shape_obj;
    if (!PyArg_ParseTuple(args, "OO", &obj, &shape_obj)) return NULL;
    NdArrayObject *a = to_ndarray(obj);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    Py_ssize_t target[NPL_MAXDIM]; int tndim;
    if (_parse_shape_arg(shape_obj, target, &tndim) < 0) { Py_DECREF(a); return NULL; }
    /* Validate broadcastability */
    Py_ssize_t out_shape[NPL_MAXDIM]; int out_ndim;
    if (broadcast_shapes(a->shape, a->ndim, target, tndim, out_shape, &out_ndim) < 0) {
        Py_DECREF(a); return NULL;
    }
    for (int i = 0; i < out_ndim; i++) {
        if (out_shape[i] != target[i]) {
            Py_DECREF(a);
            PyErr_SetString(PyExc_ValueError, "cannot broadcast to target shape");
            return NULL;
        }
    }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) { Py_DECREF(a); return NULL; }
    r->ndim = out_ndim;
    r->dtype = a->dtype;
    r->data = a->data;
    memcpy(r->shape, out_shape, out_ndim * sizeof(Py_ssize_t));
    broadcast_strides(a->shape, a->strides, a->ndim, out_ndim, r->strides);
    r->size = compute_size(out_shape, out_ndim);
    r->base = a->base ? a->base : (PyObject*)a;
    Py_INCREF(r->base);
    Py_DECREF(a);
    return (PyObject*)r;
}

static PyObject *mod_squeeze(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"a", "axis", NULL};
    PyObject *obj, *axis_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &obj, &axis_obj))
        return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *a = (NdArrayObject*)obj;
    Py_ssize_t new_shape[NPL_MAXDIM], new_strides[NPL_MAXDIM];
    int new_ndim = 0;
    if (axis_obj != Py_None) {
        int ax = (int)PyLong_AsLong(axis_obj);
        if (ax < 0) ax += a->ndim;
        for (int i = 0; i < a->ndim; i++) {
            if (i == ax) {
                if (a->shape[i] != 1) {
                    PyErr_SetString(PyExc_ValueError, "cannot squeeze axis with size != 1");
                    return NULL;
                }
                continue;
            }
            new_shape[new_ndim] = a->shape[i];
            new_strides[new_ndim] = a->strides[i];
            new_ndim++;
        }
    } else {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] == 1) continue;
            new_shape[new_ndim] = a->shape[i];
            new_strides[new_ndim] = a->strides[i];
            new_ndim++;
        }
    }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = new_ndim;
    r->dtype = a->dtype;
    r->data = a->data;
    memcpy(r->shape, new_shape, new_ndim * sizeof(Py_ssize_t));
    memcpy(r->strides, new_strides, new_ndim * sizeof(Py_ssize_t));
    r->size = a->size;
    r->base = a->base ? a->base : (PyObject*)a;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyObject *mod_expand_dims(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj;
    int axis;
    if (!PyArg_ParseTuple(args, "Oi", &obj, &axis)) return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *a = (NdArrayObject*)obj;
    int new_ndim = a->ndim + 1;
    if (axis < 0) axis += new_ndim;
    if (axis < 0 || axis >= new_ndim) {
        PyErr_SetString(PyExc_ValueError, "axis out of range"); return NULL;
    }
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = new_ndim;
    r->dtype = a->dtype;
    r->data = a->data;
    r->size = a->size;
    int si = 0;
    for (int i = 0; i < new_ndim; i++) {
        if (i == axis) {
            r->shape[i] = 1;
            r->strides[i] = (i + 1 < new_ndim && si < a->ndim) ? a->strides[si] : dtype_sizes[a->dtype];
        } else {
            r->shape[i] = a->shape[si];
            r->strides[i] = a->strides[si];
            si++;
        }
    }
    r->base = a->base ? a->base : (PyObject*)a;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyObject *mod_transpose(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *axes_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|O", &obj, &axes_obj)) return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *a = (NdArrayObject*)obj;
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = a->ndim;
    r->dtype = a->dtype;
    r->data = a->data;
    r->size = a->size;
    if (axes_obj == Py_None) {
        for (int i = 0; i < a->ndim; i++) {
            r->shape[i] = a->shape[a->ndim - 1 - i];
            r->strides[i] = a->strides[a->ndim - 1 - i];
        }
    } else {
        Py_ssize_t n = PySequence_Size(axes_obj);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PySequence_GetItem(axes_obj, i);
            int ax = (int)PyLong_AsLong(item);
            Py_DECREF(item);
            if (ax < 0) ax += a->ndim;
            r->shape[i] = a->shape[ax];
            r->strides[i] = a->strides[ax];
        }
    }
    r->base = a->base ? a->base : (PyObject*)a;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyObject *mod_swapaxes(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj;
    int ax1, ax2;
    if (!PyArg_ParseTuple(args, "Oii", &obj, &ax1, &ax2)) return NULL;
    if (!PyObject_TypeCheck(obj, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *a = (NdArrayObject*)obj;
    if (ax1 < 0) ax1 += a->ndim;
    if (ax2 < 0) ax2 += a->ndim;
    NdArrayObject *r = PyObject_New(NdArrayObject, &NdArrayType);
    if (!r) return NULL;
    r->ndim = a->ndim;
    r->dtype = a->dtype;
    r->data = a->data;
    r->size = a->size;
    memcpy(r->shape, a->shape, a->ndim * sizeof(Py_ssize_t));
    memcpy(r->strides, a->strides, a->ndim * sizeof(Py_ssize_t));
    Py_ssize_t tmp;
    tmp = r->shape[ax1]; r->shape[ax1] = r->shape[ax2]; r->shape[ax2] = tmp;
    tmp = r->strides[ax1]; r->strides[ax1] = r->strides[ax2]; r->strides[ax2] = tmp;
    r->base = a->base ? a->base : (PyObject*)a;
    Py_INCREF(r->base);
    return (PyObject*)r;
}

static PyObject *mod_concatenate(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"arrays", "axis", NULL};
    PyObject *seq;
    int axis = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &seq, &axis))
        return NULL;
    Py_ssize_t n = PySequence_Size(seq);
    if (n <= 0) { PyErr_SetString(PyExc_ValueError, "need at least one array"); return NULL; }
    PyObject *first = PySequence_GetItem(seq, 0);
    if (!first || !PyObject_TypeCheck(first, &NdArrayType)) {
        Py_XDECREF(first);
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    NdArrayObject *f = (NdArrayObject*)first;
    if (axis < 0) axis += f->ndim;
    Py_ssize_t out_shape[NPL_MAXDIM];
    memcpy(out_shape, f->shape, f->ndim * sizeof(Py_ssize_t));
    int out_ndim = f->ndim;
    int dt = f->dtype;
    Py_ssize_t total_axis = f->shape[axis];
    for (Py_ssize_t i = 1; i < n; i++) {
        PyObject *item = PySequence_GetItem(seq, i);
        NdArrayObject *a = (NdArrayObject*)item;
        dt = promote_dtype(dt, a->dtype);
        total_axis += a->shape[axis];
        Py_DECREF(item);
    }
    out_shape[axis] = total_axis;
    NdArrayObject *r = ndarray_new_empty(dt, out_shape, out_ndim);
    if (!r) { Py_DECREF(first); return NULL; }
    Py_ssize_t axis_offset = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(seq, i);
        NdArrayObject *a = (NdArrayObject*)item;
        Py_ssize_t aidx[NPL_MAXDIM];
        memset(aidx, 0, sizeof(aidx));
        for (Py_ssize_t j = 0; j < a->size; j++) {
            Py_ssize_t src_off = 0;
            for (int d = 0; d < a->ndim; d++) src_off += aidx[d]*a->strides[d];
            /* Map to output index */
            Py_ssize_t out_idx[NPL_MAXDIM];
            memcpy(out_idx, aidx, a->ndim * sizeof(Py_ssize_t));
            out_idx[axis] += axis_offset;
            Py_ssize_t dst_off = 0;
            for (int d = 0; d < out_ndim; d++) dst_off += out_idx[d]*r->strides[d];
            set_element(r->data, dt, dst_off,
                        get_element(a->data, a->dtype, src_off));
            for (int d = a->ndim - 1; d >= 0; d--) {
                if (++aidx[d] < a->shape[d]) break;
                aidx[d] = 0;
            }
        }
        axis_offset += a->shape[axis];
        Py_DECREF(item);
    }
    Py_DECREF(first);
    return (PyObject*)r;
}

static PyObject *mod_stack(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    static char *kwlist[] = {"arrays", "axis", NULL};
    PyObject *seq;
    int axis = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|i", kwlist, &seq, &axis))
        return NULL;
    Py_ssize_t n = PySequence_Size(seq);
    if (n <= 0) { PyErr_SetString(PyExc_ValueError, "need at least one array"); return NULL; }
    /* expand_dims each, then concatenate */
    PyObject *expanded = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(seq, i);
        PyObject *ea = mod_expand_dims(NULL, Py_BuildValue("(Oi)", item, axis));
        Py_DECREF(item);
        if (!ea) { Py_DECREF(expanded); return NULL; }
        PyList_SET_ITEM(expanded, i, ea);
    }
    PyObject *cat_args = Py_BuildValue("(Oi)", expanded, axis);
    PyObject *result = mod_concatenate(NULL, cat_args, NULL);
    Py_DECREF(cat_args);
    Py_DECREF(expanded);
    return result;
}

static PyObject *mod_shape(PyObject *self, PyObject *arg) {
    (void)self;
    if (!PyObject_TypeCheck(arg, &NdArrayType)) {
        PyErr_SetString(PyExc_TypeError, "expected ndarray"); return NULL;
    }
    return NdArray_get_shape((NdArrayObject*)arg, NULL);
}

static PyObject *mod_argwhere(PyObject *self, PyObject *arg) {
    (void)self;
    NdArrayObject *a = to_ndarray(arg);
    if (!a) { PyErr_SetString(PyExc_TypeError, "expected array"); return NULL; }
    /* Count nonzero */
    Py_ssize_t count = 0;
    Py_ssize_t idx[NPL_MAXDIM];
    memset(idx, 0, sizeof(idx));
    for (Py_ssize_t i = 0; i < a->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < a->ndim; d++) off += idx[d]*a->strides[d];
        if (get_element(a->data, a->dtype, off) != 0.0) count++;
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }
    int cols = a->ndim > 0 ? a->ndim : 1;
    Py_ssize_t out_shape[2] = {count, cols};
    NdArrayObject *r = ndarray_new_empty(DTYPE_INT64, out_shape, 2);
    if (!r) { Py_DECREF(a); return NULL; }
    int es = dtype_sizes[DTYPE_INT64];
    memset(idx, 0, sizeof(idx));
    Py_ssize_t ri = 0;
    for (Py_ssize_t i = 0; i < a->size; i++) {
        Py_ssize_t off = 0;
        for (int d = 0; d < a->ndim; d++) off += idx[d]*a->strides[d];
        if (get_element(a->data, a->dtype, off) != 0.0) {
            for (int d = 0; d < cols; d++)
                set_element(r->data, DTYPE_INT64, (ri * cols + d) * es, (double)idx[d]);
            ri++;
        }
        for (int d = a->ndim - 1; d >= 0; d--) {
            if (++idx[d] < a->shape[d]) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(a);
    return (PyObject*)r;
}

/* ── Section 20: Module methods table ────────────────────────────────── */

static PyMethodDef module_methods[] = {
    {"array",      (PyCFunction)mod_array,      METH_VARARGS|METH_KEYWORDS, NULL},
    {"asarray",    (PyCFunction)mod_asarray,     METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros",      (PyCFunction)mod_zeros,       METH_VARARGS|METH_KEYWORDS, NULL},
    {"ones",       (PyCFunction)mod_ones,        METH_VARARGS|METH_KEYWORDS, NULL},
    {"full",       (PyCFunction)mod_full,        METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty",      (PyCFunction)mod_empty,       METH_VARARGS|METH_KEYWORDS, NULL},
    {"empty_like", (PyCFunction)mod_empty_like,  METH_VARARGS|METH_KEYWORDS, NULL},
    {"zeros_like", (PyCFunction)mod_zeros_like,  METH_VARARGS|METH_KEYWORDS, NULL},
    {"ones_like",  (PyCFunction)mod_ones_like,   METH_VARARGS|METH_KEYWORDS, NULL},
    {"arange",     (PyCFunction)mod_arange,      METH_VARARGS|METH_KEYWORDS, NULL},
    {"linspace",   (PyCFunction)mod_linspace,    METH_VARARGS|METH_KEYWORDS, NULL},
    {"exp",        (PyCFunction)mod_exp,         METH_O,       NULL},
    {"log",        (PyCFunction)mod_log,         METH_O,       NULL},
    {"sqrt",       (PyCFunction)mod_sqrt,        METH_O,       NULL},
    {"tanh",       (PyCFunction)mod_tanh,        METH_O,       NULL},
    {"abs",        (PyCFunction)mod_abs,         METH_O,       NULL},
    {"clip",       (PyCFunction)mod_clip,        METH_VARARGS, NULL},
    {"maximum",    (PyCFunction)mod_maximum,     METH_VARARGS, NULL},
    {"minimum",    (PyCFunction)mod_minimum,     METH_VARARGS, NULL},
    {"where",      (PyCFunction)mod_where,       METH_VARARGS, NULL},
    {"sum",        (PyCFunction)mod_sum,         METH_VARARGS|METH_KEYWORDS, NULL},
    {"max",        (PyCFunction)mod_max,         METH_VARARGS|METH_KEYWORDS, NULL},
    {"min",        (PyCFunction)mod_min,         METH_VARARGS|METH_KEYWORDS, NULL},
    {"mean",       (PyCFunction)mod_mean,        METH_VARARGS|METH_KEYWORDS, NULL},
    {"var",        (PyCFunction)mod_var,         METH_VARARGS|METH_KEYWORDS, NULL},
    {"argmax",     (PyCFunction)mod_argmax,      METH_VARARGS|METH_KEYWORDS, NULL},
    {"argmin",     (PyCFunction)mod_argmin,      METH_VARARGS|METH_KEYWORDS, NULL},
    {"cumsum",     (PyCFunction)mod_cumsum,      METH_VARARGS|METH_KEYWORDS, NULL},
    {"isnan",      (PyCFunction)mod_isnan,       METH_O,       NULL},
    {"isinf",      (PyCFunction)mod_isinf,       METH_O,       NULL},
    {"isposinf",   (PyCFunction)mod_isposinf,    METH_O,       NULL},
    {"isneginf",   (PyCFunction)mod_isneginf,    METH_O,       NULL},
    {"logical_not",(PyCFunction)mod_logical_not, METH_O,       NULL},
    {"logical_and",(PyCFunction)mod_logical_and, METH_VARARGS, NULL},
    {"logical_or", (PyCFunction)mod_logical_or,  METH_VARARGS, NULL},
    {"logical_xor",(PyCFunction)mod_logical_xor, METH_VARARGS, NULL},
    {"allclose",   (PyCFunction)mod_allclose,    METH_VARARGS|METH_KEYWORDS, NULL},
    {"array_equal",(PyCFunction)mod_array_equal, METH_VARARGS, NULL},
    {"copyto",     (PyCFunction)mod_copyto,      METH_VARARGS, NULL},
    {"ascontiguousarray", (PyCFunction)mod_ascontiguousarray, METH_VARARGS, NULL},
    {"broadcast_to",     (PyCFunction)mod_broadcast_to,      METH_VARARGS, NULL},
    {"squeeze",    (PyCFunction)mod_squeeze,     METH_VARARGS|METH_KEYWORDS, NULL},
    {"expand_dims",(PyCFunction)mod_expand_dims, METH_VARARGS, NULL},
    {"transpose",  (PyCFunction)mod_transpose,   METH_VARARGS, NULL},
    {"swapaxes",   (PyCFunction)mod_swapaxes,    METH_VARARGS, NULL},
    {"stack",      (PyCFunction)mod_stack,       METH_VARARGS|METH_KEYWORDS, NULL},
    {"concatenate",(PyCFunction)mod_concatenate, METH_VARARGS|METH_KEYWORDS, NULL},
    {"shape",      (PyCFunction)mod_shape,       METH_O,       NULL},
    {"argwhere",   (PyCFunction)mod_argwhere,    METH_O,       NULL},
    {NULL, NULL, 0, NULL}
};

/* ── Section 21: Module init ─────────────────────────────────────────── */

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "numpy._core",
    .m_size    = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject *m = PyModule_Create(&module_def);
    if (!m) return NULL;

    if (PyType_Ready(&DtypeType) < 0) return NULL;
    if (PyType_Ready(&NdArrayType) < 0) return NULL;

    Py_INCREF(&NdArrayType);
    PyModule_AddObject(m, "ndarray", (PyObject*)&NdArrayType);
    Py_INCREF(&DtypeType);
    PyModule_AddObject(m, "dtype", (PyObject*)&DtypeType);

    for (int i = 0; i < DTYPE_COUNT; i++) {
        DtypeObject *d = PyObject_New(DtypeObject, &DtypeType);
        d->dtype_enum = i;
        dtype_singletons[i] = d;
        Py_INCREF(d);
        PyModule_AddObject(m, dtype_names[i], (PyObject*)d);
    }

    /* Add special constants */
    PyModule_AddObject(m, "inf", PyFloat_FromDouble(INFINITY));
    PyModule_AddObject(m, "nan", PyFloat_FromDouble(NAN));
    PyModule_AddObject(m, "pi",  PyFloat_FromDouble(3.14159265358979323846));
    PyModule_AddObject(m, "e",   PyFloat_FromDouble(2.71828182845904523536));
    PyModule_AddObject(m, "newaxis", Py_None); Py_INCREF(Py_None);

    return m;
}
