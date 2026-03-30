# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TQuant sample – INT8_SYM and INT8_ASYM quantization kernels.

INT8_SYM:   tquant(src_f32, fp_f32) -> dst_i8
INT8_ASYM:  tquant(src_f32, fp_f32, offset_f32) -> dst_ui8

Each kernel:
  1. Loads a 16x16 f32 tile from global memory.
  2. Loads a 16x16 f32 scaling-factor (fp) tile from global memory.
  3. For ASYM: loads a 16x16 f32 offset tile from global memory.
  4. Calls pto.tquant.
  5. Stores the 16x16 i8 / ui8 result tile to global memory.
"""

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    F32Type, IndexType, IntegerType,
)
from mlir.dialects import func, arith, pto


# Tile shape used throughout the sample.
_SHAPE = [16, 16]


def _make_common_types(ctx):
    """Return a namespace of commonly used types / attrs."""
    f32  = F32Type.get(ctx)
    i8   = IntegerType.get_signless(8, ctx)
    ui8  = IntegerType.get_unsigned(8, ctx)
    idx  = IndexType.get(ctx)

    ptr_f32  = pto.PtrType.get(f32, ctx)
    ptr_i8   = pto.PtrType.get(i8, ctx)
    ptr_ui8  = pto.PtrType.get(ui8, ctx)

    tv2_f32  = pto.TensorViewType.get(2, f32, ctx)
    tv2_i8   = pto.TensorViewType.get(2, i8, ctx)
    tv2_ui8  = pto.TensorViewType.get(2, ui8, ctx)

    ptv_f32  = pto.PartitionTensorViewType.get(_SHAPE, f32, ctx)
    ptv_i8   = pto.PartitionTensorViewType.get(_SHAPE, i8, ctx)
    ptv_ui8  = pto.PartitionTensorViewType.get(_SHAPE, ui8, ctx)

    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    bl  = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
    sl  = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
    pd  = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
    cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

    tb_f32  = pto.TileBufType.get(_SHAPE, f32, vec, _SHAPE, cfg, ctx)
    tb_i8   = pto.TileBufType.get(_SHAPE, i8,  vec, _SHAPE, cfg, ctx)
    tb_ui8  = pto.TileBufType.get(_SHAPE, ui8, vec, _SHAPE, cfg, ctx)

    quant_sym  = pto.QuantTypeAttr.get(pto.QuantType.INT8_SYM,  ctx)
    quant_asym = pto.QuantTypeAttr.get(pto.QuantType.INT8_ASYM, ctx)

    class NS: pass
    ns = NS()
    ns.f32 = f32; ns.i8 = i8; ns.ui8 = ui8; ns.idx = idx
    ns.ptr_f32 = ptr_f32; ns.ptr_i8 = ptr_i8; ns.ptr_ui8 = ptr_ui8
    ns.tv2_f32 = tv2_f32; ns.tv2_i8 = tv2_i8; ns.tv2_ui8 = tv2_ui8
    ns.ptv_f32 = ptv_f32; ns.ptv_i8 = ptv_i8; ns.ptv_ui8 = ptv_ui8
    ns.tb_f32 = tb_f32; ns.tb_i8 = tb_i8; ns.tb_ui8 = tb_ui8
    ns.quant_sym = quant_sym; ns.quant_asym = quant_asym
    return ns


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            t = _make_common_types(ctx)

            # ------------------------------------------------------------------
            # @tquant_sym_kernel(src_ptr: !pto.ptr<f32>,
            #                    fp_ptr:  !pto.ptr<f32>,
            #                    dst_ptr: !pto.ptr<i8>)
            # ------------------------------------------------------------------
            fn_sym_ty = func.FunctionType.get([t.ptr_f32, t.ptr_f32, t.ptr_i8], [])
            with InsertionPoint(m.body):
                fn_sym = func.FuncOp("tquant_sym_kernel", fn_sym_ty)
                entry_sym = fn_sym.add_entry_block()

            with InsertionPoint(entry_sym):
                idx = t.idx
                c0  = arith.ConstantOp(idx, 0).result
                c1  = arith.ConstantOp(idx, 1).result
                c16 = arith.ConstantOp(idx, 16).result

                src_ptr, fp_ptr, dst_ptr = entry_sym.arguments

                # Make tensor views over the flat global-memory pointers.
                tv_src = pto.MakeTensorViewOp(t.tv2_f32, src_ptr, [c16, c16], [c16, c1]).result
                tv_fp  = pto.MakeTensorViewOp(t.tv2_f32, fp_ptr,  [c16, c16], [c16, c1]).result
                tv_dst = pto.MakeTensorViewOp(t.tv2_i8,  dst_ptr, [c16, c16], [c16, c1]).result

                # Partition into tile-sized sub-views.
                sv_src = pto.PartitionViewOp(t.ptv_f32, tv_src, offsets=[c0, c0], sizes=[c16, c16]).result
                sv_fp  = pto.PartitionViewOp(t.ptv_f32, tv_fp,  offsets=[c0, c0], sizes=[c16, c16]).result
                sv_dst = pto.PartitionViewOp(t.ptv_i8,  tv_dst, offsets=[c0, c0], sizes=[c16, c16]).result

                # Allocate on-chip tile buffers.
                tb_src = pto.AllocTileOp(t.tb_f32).result
                tb_fp  = pto.AllocTileOp(t.tb_f32).result
                tb_dst = pto.AllocTileOp(t.tb_i8).result

                # Load src and fp tiles from global memory.
                pto.TLoadOp(None, sv_src, tb_src)
                pto.TLoadOp(None, sv_fp,  tb_fp)

                # INT8_SYM quantization (no offset operand).
                pto.TQuantOp(tb_src, tb_fp, tb_dst, quant_type=t.quant_sym)

                # Store result back to global memory.
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            # ------------------------------------------------------------------
            # @tquant_asym_kernel(src_ptr:    !pto.ptr<f32>,
            #                     fp_ptr:     !pto.ptr<f32>,
            #                     offset_ptr: !pto.ptr<f32>,
            #                     dst_ptr:    !pto.ptr<ui8>)
            # ------------------------------------------------------------------
            fn_asym_ty = func.FunctionType.get(
                [t.ptr_f32, t.ptr_f32, t.ptr_f32, t.ptr_ui8], [])
            with InsertionPoint(m.body):
                fn_asym = func.FuncOp("tquant_asym_kernel", fn_asym_ty)
                entry_asym = fn_asym.add_entry_block()

            with InsertionPoint(entry_asym):
                idx = t.idx
                c0  = arith.ConstantOp(idx, 0).result
                c1  = arith.ConstantOp(idx, 1).result
                c16 = arith.ConstantOp(idx, 16).result

                src_ptr, fp_ptr, off_ptr, dst_ptr = entry_asym.arguments

                # Make tensor views.
                tv_src = pto.MakeTensorViewOp(t.tv2_f32, src_ptr, [c16, c16], [c16, c1]).result
                tv_fp  = pto.MakeTensorViewOp(t.tv2_f32, fp_ptr,  [c16, c16], [c16, c1]).result
                tv_off = pto.MakeTensorViewOp(t.tv2_f32, off_ptr, [c16, c16], [c16, c1]).result
                tv_dst = pto.MakeTensorViewOp(t.tv2_ui8, dst_ptr, [c16, c16], [c16, c1]).result

                # Partition into sub-views.
                sv_src = pto.PartitionViewOp(t.ptv_f32, tv_src, offsets=[c0, c0], sizes=[c16, c16]).result
                sv_fp  = pto.PartitionViewOp(t.ptv_f32, tv_fp,  offsets=[c0, c0], sizes=[c16, c16]).result
                sv_off = pto.PartitionViewOp(t.ptv_f32, tv_off, offsets=[c0, c0], sizes=[c16, c16]).result
                sv_dst = pto.PartitionViewOp(t.ptv_ui8, tv_dst, offsets=[c0, c0], sizes=[c16, c16]).result

                # Allocate on-chip tile buffers.
                tb_src = pto.AllocTileOp(t.tb_f32).result
                tb_fp  = pto.AllocTileOp(t.tb_f32).result
                tb_off = pto.AllocTileOp(t.tb_f32).result
                tb_dst = pto.AllocTileOp(t.tb_ui8).result

                # Load tiles from global memory.
                pto.TLoadOp(None, sv_src, tb_src)
                pto.TLoadOp(None, sv_fp,  tb_fp)
                pto.TLoadOp(None, sv_off, tb_off)

                # INT8_ASYM quantization (offset operand required).
                pto.TQuantOp(tb_src, tb_fp, tb_dst,
                             quant_type=t.quant_asym,
                             offset=tb_off)

                # Store result back to global memory.
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
