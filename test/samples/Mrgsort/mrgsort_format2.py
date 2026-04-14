# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TMrgSortOp format2: ins(src0..srcN) outs(dst, tmp, excuted) with exhausted attr,
where N is 1, 2, or 3 (2-way / 3-way / 4-way merge).

Important notes for on-device execution:
  - PTOAS now supports 2-way, 3-way, and 4-way merge forms.
  - This testcase covers all three forms in a single function entry so it fits
    the remote validation flow.
"""
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType, IntegerType

from mlir.ir import VectorType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            i16 = IntegerType.get_signless(16, ctx)
            # Format2 excuted: vector<4xi16>
            vec_4_i16 = VectorType.get([4], i16)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            # Each list holds packed (value, index) structures. In pto-isa each
            # structure is 8 bytes (2 x f32 when using float tiles). Keep the
            # testcase small enough to satisfy device constraints across SoCs.
            #
            # 4 lists * 64 structures/list = 256 structures output.
            # 1 structure = 2 floats => 128 floats/list, 512 floats output.
            part_view_1x128 = pto.PartitionTensorViewType.get([1, 128], f32, ctx)
            part_view_1x256 = pto.PartitionTensorViewType.get([1, 256], f32, ctx)
            part_view_1x384 = pto.PartitionTensorViewType.get([1, 384], f32, ctx)
            part_view_1x512 = pto.PartitionTensorViewType.get([1, 512], f32, ctx)
            part_view_1x1152 = pto.PartitionTensorViewType.get([1, 1152], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_1x256 = pto.TileBufType.get([1, 256], f32, vec, [1, 256], cfg, ctx)
            tile_buf_1x384 = pto.TileBufType.get([1, 384], f32, vec, [1, 384], cfg, ctx)
            tile_buf_1x128 = pto.TileBufType.get([1, 128], f32, vec, [1, 128], cfg, ctx)
            tile_buf_1x512 = pto.TileBufType.get([1, 512], f32, vec, [1, 512], cfg, ctx)

            # Kernel: (in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr, executed_list) -> ()
            fn_ty = func.FunctionType.get(
                [ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32, vec_4_i16], []
            )
            with InsertionPoint(m.body):
                fn = func.FuncOp("mrgsort_format2_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c128 = arith.ConstantOp(IndexType.get(ctx), 128).result
                c256 = arith.ConstantOp(IndexType.get(ctx), 256).result
                c384 = arith.ConstantOp(IndexType.get(ctx), 384).result
                c512 = arith.ConstantOp(IndexType.get(ctx), 512).result
                c640 = arith.ConstantOp(IndexType.get(ctx), 640).result
                c1152 = arith.ConstantOp(IndexType.get(ctx), 1152).result

                arg0, arg1, arg2, arg3, arg_out, excuted = entry.arguments

                # Inputs: 1x128 (128 f32) per list.
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c1, c128], [c128, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c1, c128], [c128, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c1, c128], [c128, c1]).result
                tv3 = pto.MakeTensorViewOp(tv2_f32, arg3, [c1, c128], [c128, c1]).result
                # Output buffer holds:
                #   2-way result: 1x256
                #   3-way result: 1x384
                #   4-way result: 1x512
                # total = 1152 f32
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c1, c1152], [c1152, c1]).result

                sv0 = pto.PartitionViewOp(part_view_1x128, tv0, offsets=[c0, c0], sizes=[c1, c128]).result
                sv1 = pto.PartitionViewOp(part_view_1x128, tv1, offsets=[c0, c0], sizes=[c1, c128]).result
                sv2 = pto.PartitionViewOp(part_view_1x128, tv2, offsets=[c0, c0], sizes=[c1, c128]).result
                sv3 = pto.PartitionViewOp(part_view_1x128, tv3, offsets=[c0, c0], sizes=[c1, c128]).result

                # Format2 source tiles shared by all 2-way / 3-way / 4-way cases.
                tb_s0 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s1 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s2 = pto.AllocTileOp(tile_buf_1x128).result
                tb_s3 = pto.AllocTileOp(tile_buf_1x128).result
                tb_dst2 = pto.AllocTileOp(tile_buf_1x256).result
                tb_tmp2 = pto.AllocTileOp(tile_buf_1x256).result
                tb_dst3 = pto.AllocTileOp(tile_buf_1x384).result
                tb_tmp3 = pto.AllocTileOp(tile_buf_1x384).result
                tb_dst4 = pto.AllocTileOp(tile_buf_1x512).result
                tb_tmp4 = pto.AllocTileOp(tile_buf_1x512).result

                pto.TLoadOp(None, sv0, tb_s0)
                pto.TLoadOp(None, sv1, tb_s1)
                pto.TLoadOp(None, sv2, tb_s2)
                pto.TLoadOp(None, sv3, tb_s3)

                # 2-way: src0 + src1 -> 1x256
                pto.TMrgSortOp(
                    srcs=[tb_s0, tb_s1],
                    dsts=[tb_dst2, tb_tmp2],
                    excuted=excuted,
                    exhausted=False,
                )

                # 3-way: src0 + src1 + src2 -> 1x384
                pto.TMrgSortOp(
                    srcs=[tb_s0, tb_s1, tb_s2],
                    dsts=[tb_dst3, tb_tmp3],
                    excuted=excuted,
                    exhausted=False,
                )

                # 4-way: src0 + src1 + src2 + src3 -> 1x512
                pto.TMrgSortOp(
                    srcs=[tb_s0, tb_s1, tb_s2, tb_s3],
                    dsts=[tb_dst4, tb_tmp4],
                    excuted=excuted,
                    exhausted=True,
                )

                sv_out2 = pto.PartitionViewOp(part_view_1x256, tv_out, offsets=[c0, c0], sizes=[c1, c256]).result
                sv_out3 = pto.PartitionViewOp(part_view_1x384, tv_out, offsets=[c0, c256], sizes=[c1, c384]).result
                sv_out4 = pto.PartitionViewOp(part_view_1x512, tv_out, offsets=[c0, c640], sizes=[c1, c512]).result
                pto.TStoreOp(None, tb_dst2, sv_out2)
                pto.TStoreOp(None, tb_dst3, sv_out3)
                pto.TStoreOp(None, tb_dst4, sv_out4)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
