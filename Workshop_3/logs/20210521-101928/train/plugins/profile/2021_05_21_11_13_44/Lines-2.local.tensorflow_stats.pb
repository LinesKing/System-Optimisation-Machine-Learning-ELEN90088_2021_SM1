"?g
BHostIDLE"IDLE1     W?@A     W?@a??yr͎??i??yr͎???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ؄@9     ؄@A     ؄@I     ؄@ae#S?????iP?#J?????Unknown?
vHost_FusedMatMul"sequential_66/dense_205/Relu(1     ?M@9     ?M@A     ?M@I     ?M@a??e[??z?i,??R;"???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?G@9     ?G@A      E@I      E@a|R??s?i??ijaH???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@aW?r?3?h?i?R!??`???Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@a]5x??g?i?0K?x???Unknown?
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a7?X??]?irw??????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      ,@9      ,@A      ,@I      ,@aP?m_?nY?iT.?œ???Unknown
u	HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@aP?m_?nY?i6?p|????Unknown
?
HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      ,@9      ,@A      ,@I      ,@aP?m_?nY?i???3????Unknown
VHostMean"Mean(1      (@9      (@A      (@I      (@ai?????U?il]????Unknown
eHost
LogicalAnd"
LogicalAnd(1      &@9      &@A      &@I      &@av?o??S?i?#W?????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      $@9      $@A      $@I      $@a????*R?i???(-????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      3@9      3@A      $@I      $@a????*R?i??vkB????Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a????*R?iK??W????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a????*R?iS??l????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      $@9      $@A      $@I      $@a????*R?i?&3?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a????wYP?i????????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      "@9      "@A      "@I      "@a????wYP?i?????????Unknown
?HostMatMul".gradient_tape/sequential_66/dense_206/MatMul_1(1      "@9      "@A      "@I      "@a????wYP?iԑ?f???Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a7?X??M?ih?L???Unknown
dHostDataset"Iterator::Model(1     ?A@9     ?A@A       @I       @a7?X??M?iD>]ѐ???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a7?X??M?i|?????Unknown
?HostMatMul",gradient_tape/sequential_66/dense_206/MatMul(1       @9       @A       @I       @a7?X??M?i???;%???Unknown
vHost_FusedMatMul"sequential_66/dense_206/Relu(1       @9       @A       @I       @a7?X??M?i??q],???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a7?X??M?i$?\??3???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aP?m_?nI?i?r?T?9???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aP?m_?nI?iN?Y@???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aP?m_?nI?iw)$??F???Unknown
?HostMatMul",gradient_tape/sequential_66/dense_205/MatMul(1      @9      @A      @I      @aP?m_?nI?i??`M???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_66/dense_207/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aP?m_?nI?iY?SlS???Unknown
? HostMatMul",gradient_tape/sequential_66/dense_207/MatMul(1      @9      @A      @I      @aP?m_?nI?iʻ???Y???Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?R@9     ?R@A      @I      @ai?????E?it???:_???Unknown
?"HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ai?????E?i}??d???Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a????*B?ic?8i???Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a????*B?i?H[P?m???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a????*B?i?.??Mr???Unknown
V&HostSum"Sum_2(1      @9      @A      @I      @a????*B?i????v???Unknown
v'HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a????*B?i??24c{???Unknown
?(HostBiasAddGrad"9gradient_tape/sequential_66/dense_206/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????*B?ip?z?????Unknown
s)HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a7?X??=?i????????Unknown
z*HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a7?X??=?i???
2????Unknown
~+HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a7?X??=?iġZ%Ԋ???Unknown
v,HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a7?X??=?i????v????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a7?X??=?i?w?Z????Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a7?X??=?ic:u?????Unknown
?/HostMatMul".gradient_tape/sequential_66/dense_207/MatMul_1(1      @9      @A      @I      @a7?X??=?i4Nڏ\????Unknown
?0HostReadVariableOp"-sequential_66/dense_206/MatMul/ReadVariableOp(1      @9      @A      @I      @a7?X??=?iP9z??????Unknown
y1Host_FusedMatMul"sequential_66/dense_207/BiasAdd(1      @9      @A      @I      @a7?X??=?il$Š????Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @ai?????5?i?YZ????Unknown
\3HostGreater"Greater(1      @9      @A      @I      @ai?????5?i
?????Unknown
?4HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @ai?????5?ik??ͨ???Unknown
r5HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @ai?????5?i????????Unknown
?6HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @ai?????5?i???@????Unknown
v7HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @ai?????5?ij??<?????Unknown
?8HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @ai?????5?i???г????Unknown
`9HostDivNoNan"
div_no_nan(1      @9      @A      @I      @ai?????5?i??dm????Unknown
b:HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @ai?????5?ii???&????Unknown
~;HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @ai?????5?i??Ɍ?????Unknown
?<HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @ai?????5?ix? ?????Unknown
?=HostBiasAddGrad"9gradient_tape/sequential_66/dense_205/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ai?????5?ihh??S????Unknown
t>HostSigmoid"sequential_66/dense_207/Sigmoid(1      @9      @A      @I      @ai?????5?i?X?H????Unknown
v?HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a7?X??-?iKNV?????Unknown
V@HostCast"Cast(1       @9       @A       @I       @a7?X??-?i?CQc?????Unknown
XAHostCast"Cast_3(1       @9       @A       @I       @a7?X??-?ig9?p?????Unknown
XBHostEqual"Equal(1       @9       @A       @I       @a7?X??-?i?.?}Q????Unknown
aCHostIdentity"Identity(1       @9       @A       @I       @a7?X??-?i?$A?"????Unknown?
?DHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A       @I       @a7?X??-?i???????Unknown
dEHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a7?X??-?i????????Unknown
jFHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a7?X??-?i-1??????Unknown
vGHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a7?X??-?i????f????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a7?X??-?iI???7????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a7?X??-?i?? ?????Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a7?X??-?ie?p??????Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a7?X??-?i?????????Unknown
xLHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a7?X??-?i??|????Unknown
?MHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a7?X??-?i?`M????Unknown
?NHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a7?X??-?i???????Unknown
?OHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a7?X??-?i+? +?????Unknown
?PHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a7?X??-?i??P8?????Unknown
?QHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a7?X??-?iG??E?????Unknown
~RHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a7?X??-?iՇ?Rb????Unknown
?SHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a7?X??-?ic}@`3????Unknown
?THostReluGrad".gradient_tape/sequential_66/dense_205/ReluGrad(1       @9       @A       @I       @a7?X??-?i?r?m????Unknown
?UHostReadVariableOp"-sequential_66/dense_205/MatMul/ReadVariableOp(1       @9       @A       @I       @a7?X??-?ih?z?????Unknown
?VHostReadVariableOp".sequential_66/dense_206/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a7?X??-?i^0??????Unknown
?WHostReadVariableOp"-sequential_66/dense_207/MatMul/ReadVariableOp(1       @9       @A       @I       @a7?X??-?i?S??w????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a7?X???ibN(`????Unknown
XYHostCast"Cast_4(1      ??9      ??A      ??I      ??a7?X???i)IТH????Unknown
XZHostCast"Cast_5(1      ??9      ??A      ??I      ??a7?X???i?Cx)1????Unknown
T[HostMul"Mul(1      ??9      ??A      ??I      ??a7?X???i?> ?????Unknown
|\HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a7?X???i~9?6????Unknown
w]HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7?X???iE4p??????Unknown
y^HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a7?X???i/D?????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a7?X???i?)?ʻ????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a7?X???i?$hQ?????Unknown
?aHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a7?X???ia،????Unknown
?bHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a7?X???i(?^u????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a7?X???i?`?]????Unknown
?dHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a7?X???i?lF????Unknown
?eHostReluGrad".gradient_tape/sequential_66/dense_206/ReluGrad(1      ??9      ??A      ??I      ??a7?X???i}
??.????Unknown
?fHostReadVariableOp".sequential_66/dense_205/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a7?X???iDXy????Unknown
?gHostReadVariableOp".sequential_66/dense_207/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a7?X???i     ???Unknown
WhHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i     ???Unknown
YiHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i     ???Unknown
[jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i     ???Unknown*?g
uHostFlushSummaryWriter"FlushSummaryWriter(1     ؄@9     ؄@A     ؄@I     ؄@ae???D??ie???D???Unknown?
vHost_FusedMatMul"sequential_66/dense_205/Relu(1     ?M@9     ?M@A     ?M@I     ?M@ad?)5? ??i??B'P????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1     ?G@9     ?G@A      E@I      E@a?Z??,??ix???????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@a'P??C??i?G??Q???Unknown
iHostWriteSummary"WriteSummary(1      :@9      :@A      :@I      :@aB'P??C??i3?Z?????Unknown?
^HostGatherV2"GatherV2(1      0@9      0@A      0@I      0@a?Ň*,??i??yÙd???Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      ,@9      ,@A      ,@I      ,@ay?6????i?UX4????Unknown
uHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ,@9      ,@A      ,@I      ,@ay?6????inA0?????Unknown
?	HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      ,@9      ,@A      ,@I      ,@ay?6????iRs?iw???Unknown
V
HostMean"Mean(1      (@9      (@A      (@I      (@ax??????i?¢?????Unknown
eHost
LogicalAnd"
LogicalAnd(1      &@9      &@A      &@I      &@a??W=]???i? ?v????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      $@9      $@A      $@I      $@a?1۔?[??i???UO???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      3@9      3@A      $@I      $@a?1۔?[??iG?>KĐ???Unknown
lHostIteratorGetNext"IteratorGetNext(1      $@9      $@A      $@I      $@a?1۔?[??ig??2????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      $@9      $@A      $@I      $@a?1۔?[??i???????Unknown
vHostExp"%binary_crossentropy/logistic_loss/Exp(1      $@9      $@A      $@I      $@a?1۔?[??i?@9?U???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      "@9      "@A      "@I      "@a4??دq}?i????????Unknown
?Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      "@9      "@A      "@I      "@a4??دq}?i?7?I?????Unknown
?HostMatMul".gradient_tape/sequential_66/dense_206/MatMul_1(1      "@9      "@A      "@I      "@a4??دq}?i?M?????Unknown
`HostGatherV2"
GatherV2_1(1       @9       @A       @I       @a?Ň*,z?iX=]?:???Unknown
dHostDataset"Iterator::Model(1     ?A@9     ?A@A       @I       @a?Ň*,z?i??lSjn???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?Ň*,z?i?Q|?¢???Unknown
?HostMatMul",gradient_tape/sequential_66/dense_206/MatMul(1       @9       @A       @I       @a?Ň*,z?i܋?????Unknown
vHost_FusedMatMul"sequential_66/dense_206/Relu(1       @9       @A       @I       @a?Ň*,z?i<f?Rs???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?Ň*,z?iu????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @ay?6??v?ig???m???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @ay?6??v?iY"?<f????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @ay?6??v?iK???3????Unknown
?HostMatMul",gradient_tape/sequential_66/dense_205/MatMul(1      @9      @A      @I      @ay?6??v?i=Ta? ????Unknown
?HostBiasAddGrad"9gradient_tape/sequential_66/dense_207/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ay?6??v?i/???$???Unknown
?HostMatMul",gradient_tape/sequential_66/dense_207/MatMul(1      @9      @A      @I      @ay?6??v?i!?<f?R???Unknown
x HostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?R@9     ?R@A      @I      @ax????s?i?-??y???Unknown
?!HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @ax????s?iw???????Unknown
?"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?1۔?[p?iۋ??????Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a?1۔?[p?i?B'P?????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?1۔?[p?i??P?E???Unknown
V%HostSum"Sum_2(1      @9      @A      @I      @a?1۔?[p?i?z??#???Unknown
v&HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?1۔?[p?ike???D???Unknown
?'HostBiasAddGrad"9gradient_tape/sequential_66/dense_206/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?1۔?[p?i??$ke???Unknown
s(HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a?Ň*,j?i??UO????Unknown
z)HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?Ň*,j?i	??yÙ???Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?Ň*,j?i&ke??????Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?Ň*,j?iC0??????Unknown
?,HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a?Ň*,j?i`?t?G????Unknown
?-HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?Ň*,j?i}??#t???Unknown
?.HostMatMul".gradient_tape/sequential_66/dense_207/MatMul_1(1      @9      @A      @I      @a?Ň*,j?i??N????Unknown
?/HostReadVariableOp"-sequential_66/dense_206/MatMul/ReadVariableOp(1      @9      @A      @I      @a?Ň*,j?i?Dy?6???Unknown
y0Host_FusedMatMul"sequential_66/dense_207/BiasAdd(1      @9      @A      @I      @a?Ň*,j?i?	???P???Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @ax????c?i??yÙd???Unknown
\2HostGreater"Greater(1      @9      @A      @I      @ax????c?i~?_?:x???Unknown
?3HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @ax????c?iS?E܋???Unknown
r4HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @ax????c?i(Y+#}????Unknown
?5HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @ax????c?i?,C????Unknown
v6HostSub"%binary_crossentropy/logistic_loss/sub(1      @9      @A      @I      @ax????c?i? ?b?????Unknown
?7HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @ax????c?i??܂`????Unknown
`8HostDivNoNan"
div_no_nan(1      @9      @A      @I      @ax????c?i|?¢????Unknown
b9HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @ax????c?iQ|?¢???Unknown
~:HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @ax????c?i&P??C???Unknown
?;HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @ax????c?i?#t?(???Unknown
?<HostBiasAddGrad"9gradient_tape/sequential_66/dense_205/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ax????c?i??Y"?<???Unknown
t=HostSigmoid"sequential_66/dense_207/Sigmoid(1      @9      @A      @I      @ax????c?i???B'P???Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?Ň*,Z?i3??W=]???Unknown
V?HostCast"Cast(1       @9       @A       @I       @a?Ň*,Z?i???lSj???Unknown
X@HostCast"Cast_3(1       @9       @A       @I       @a?Ň*,Z?iOs?iw???Unknown
XAHostEqual"Equal(1       @9       @A       @I       @a?Ň*,Z?i?UO?????Unknown
aBHostIdentity"Identity(1       @9       @A       @I       @a?Ň*,Z?ik8???????Unknown?
?CHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A       @I       @a?Ň*,Z?i????????Unknown
dDHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?Ň*,Z?i????????Unknown
jEHostMean"binary_crossentropy/Mean(1       @9       @A       @I       @a?Ň*,Z?i?^?׸???Unknown
vFHostSum"%binary_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a?Ň*,Z?i?¢?????Unknown
}GHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?Ň*,Z?i1??????Unknown
uHHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?Ň*,Z?i??*,????Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?Ň*,Z?iMjnA0????Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?Ň*,Z?i?L?VF????Unknown
xKHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?Ň*,Z?ii/?k\???Unknown
?LHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1       @9       @A       @I       @a?Ň*,Z?i?:?r???Unknown
?MHostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?Ň*,Z?i??}??!???Unknown
?NHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?Ň*,Z?i????.???Unknown
?OHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a?Ň*,Z?i????;???Unknown
?PHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?Ň*,Z?i/?I??H???Unknown
~QHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?Ň*,Z?i?~???U???Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?Ň*,Z?iKa? ?b???Unknown
?SHostReluGrad".gradient_tape/sequential_66/dense_205/ReluGrad(1       @9       @A       @I       @a?Ň*,Z?i?Cp???Unknown
?THostReadVariableOp"-sequential_66/dense_205/MatMul/ReadVariableOp(1       @9       @A       @I       @a?Ň*,Z?ig&Y+#}???Unknown
?UHostReadVariableOp".sequential_66/dense_206/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?Ň*,Z?i??@9????Unknown
?VHostReadVariableOp"-sequential_66/dense_207/MatMul/ReadVariableOp(1       @9       @A       @I       @a?Ň*,Z?i???UO????Unknown
vWHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?Ň*,J?i?܂`ڝ???Unknown
XXHostCast"Cast_4(1      ??9      ??A      ??I      ??a?Ň*,J?i?$ke????Unknown
XYHostCast"Cast_5(1      ??9      ??A      ??I      ??a?Ň*,J?iX??u?????Unknown
TZHostMul"Mul(1      ??9      ??A      ??I      ??a?Ň*,J?i??h?{????Unknown
|[HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?Ň*,J?i??
?????Unknown
w\HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?Ň*,J?i-????????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?Ň*,J?it?N?????Unknown
?^HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?Ň*,J?i?u???????Unknown
?_Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?Ň*,J?ig??2????Unknown
?`HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?Ň*,J?iIX4??????Unknown
?aHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a?Ň*,J?i?I??H????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?Ň*,J?i?:x??????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?Ň*,J?i,?^????Unknown
?dHostReluGrad".gradient_tape/sequential_66/dense_206/ReluGrad(1      ??9      ??A      ??I      ??a?Ň*,J?ie???????Unknown
?eHostReadVariableOp".sequential_66/dense_205/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?Ň*,J?i?^?t????Unknown
?fHostReadVariableOp".sequential_66/dense_207/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?Ň*,J?i?????????Unknown
WgHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown
[iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(i?????????Unknown2CPU