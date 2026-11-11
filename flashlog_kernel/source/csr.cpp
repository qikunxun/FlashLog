#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using Input = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

// Two stable counting passes: relation first, source second. No comparison sort.
py::tuple build_side(Input source, Input target, Input relation, int64_t nodes) {
    const int64_t edges = source.size();
    if (source.ndim()!=1 || target.ndim()!=1 || relation.ndim()!=1 ||
        target.size()!=edges || relation.size()!=edges || nodes<0 ||
        nodes>INT32_MAX || edges>INT32_MAX)
        throw py::value_error("CSR requires equal 1D arrays and int32-sized graph");
    const auto *s=source.data(), *t=target.data(), *r=relation.data();
    int64_t relations=0;
    for (int64_t i=0;i<edges;++i) {
        if (s[i]<0 || s[i]>=nodes || t[i]<0 || t[i]>=nodes || r[i]<0 || r[i]>INT16_MAX)
            throw py::value_error("CSR entity or relation ID outside supported range");
        relations=std::max(relations,r[i]+1);
    }
    py::array_t<int32_t> order(edges), row_ptr(nodes+1), columns(edges),
        group_ptr(nodes+1);
    py::array_t<int16_t> edge_rel(edges);
    auto *o=order.mutable_data(), *rp=row_ptr.mutable_data(),
        *c=columns.mutable_data(), *gp=group_ptr.mutable_data();
    auto *er=edge_rel.mutable_data();
    int64_t groups=0;
    {
        py::gil_scoped_release release;
        std::vector<int32_t> temporary(edges);
        std::vector<int64_t> cursor(std::max(nodes,relations),0);
        for (int64_t i=0;i<edges;++i) ++cursor[r[i]];
        int64_t offset=0;
        for (int64_t i=0;i<relations;++i) { auto count=cursor[i]; cursor[i]=offset; offset+=count; }
        for (int64_t i=0;i<edges;++i) temporary[cursor[r[i]]++]=i;
        std::fill(cursor.begin(),cursor.end(),0);
        for (int64_t i=0;i<edges;++i) ++cursor[s[i]];
        offset=0;
        for (int64_t i=0;i<nodes;++i) { auto count=cursor[i]; rp[i]=offset; cursor[i]=offset; offset+=count; }
        rp[nodes]=edges;
        for (auto i:temporary) o[cursor[s[i]]++]=i;
        std::fill(gp,gp+nodes+1,0);
        for (int64_t j=0;j<edges;++j) {
            auto i=o[j]; c[j]=t[i]; er[j]=r[i];
            if (j==0 || s[i]!=s[o[j-1]] || r[i]!=r[o[j-1]]) { ++groups; ++gp[s[i]+1]; }
        }
        for (int64_t i=0;i<nodes;++i) gp[i+1]+=gp[i];
    }
    py::array_t<int16_t> group_rel(groups);
    py::array_t<int32_t> starts(groups), counts(groups);
    auto *gr=group_rel.mutable_data(); auto *gs=starts.mutable_data(), *gc=counts.mutable_data();
    {
        py::gil_scoped_release release;
        int64_t g=-1;
        for (int64_t j=0;j<edges;++j) {
            auto i=o[j];
            if (j==0 || s[i]!=s[o[j-1]] || r[i]!=r[o[j-1]]) {
                if (g>=0) gc[g]=j-gs[g];
                ++g; gs[g]=j; gr[g]=r[i];
            }
        }
        if (g>=0) gc[g]=edges-gs[g];
    }
    return py::make_tuple(row_ptr,columns,edge_rel,order,group_ptr,group_rel,starts,counts);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("build_side", &build_side); }
