"?e
BHostIDLE"IDLE1    ???@A    ???@ak?????ik??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@aQp???i'l/?H???Unknown?
xHost_FusedMatMul"sequential_10/dense_33/BiasAdd(1      F@9      F@A      F@I      F@aN??#?9q?i?ٳ+-k???Unknown
iHostWriteSummary"WriteSummary(1     ?A@9     ?A@A     ?A@I     ?A@ae????gk?iz?x?????Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      @@9      @@A      ?@I      ?@a????Fh?iW?&۞???Unknown
uHost_FusedMatMul"sequential_10/dense_31/Relu(1      =@9      =@A      =@I      =@a??,?f?iO?7R?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@a?-?C$e?ib????????Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      5@9      5@A      5@I      5@a?cx܉q`?i?V?&????Unknown
?	HostMatMul"-gradient_tape/sequential_10/dense_32/MatMul_1(1      1@9      1@A      1@I      1@a?3U}q?Z?i`\?u????Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      .@9      .@A      .@I      .@aj?_?}W?iŋ?4????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@aj?_?}W?iʈ?x?????Unknown
HostMatMul"+gradient_tape/sequential_10/dense_31/MatMul(1      *@9      *@A      *@I      *@a-??A?[T?i?e\`!
???Unknown
dHostDataset"Iterator::Model(1      B@9      B@A      "@I      "@a?<Z0L?i?t?v-???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?Nn??I?i4q???Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?Nn??I?iȫV?????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?Nn??I?i\G?]?#???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?Nn??I?i????;*???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?Nn??I?i?~	?0???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?Nn??I?iED?6???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?Nn??I?i????=???Unknown
HostMatMul"+gradient_tape/sequential_10/dense_32/MatMul(1       @9       @A       @I       @a?Nn??I?i@Q??JC???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??з?E?iay???H???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a??з?E?i????@N???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a??з?E?i?ɘ?S???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a??з?E?i???@7Y???Unknown
HostMatMul"+gradient_tape/sequential_10/dense_33/MatMul(1      @9      @A      @I      @a??з?E?i??n?^???Unknown
uHost_FusedMatMul"sequential_10/dense_32/Relu(1      @9      @A      @I      @a??з?E?iBu?-d???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a??з?E?i'jiʨi???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a>?Ҳ??B?i??[n???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_10/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a>?Ҳ??B?i???=s???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_10/dense_32/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a>?Ҳ??B?i4?o??w???Unknown
V HostSum"Sum_2(1      @9      @A      @I      @a??	*+R??ip??<?{???Unknown
v!HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??	*+R??i?
:?????Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a??	*+R??i?K??????Unknown
?#HostBiasAddGrad"8gradient_tape/sequential_10/dense_33/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??	*+R??i$?j????Unknown
?$HostReadVariableOp",sequential_10/dense_32/MatMul/ReadVariableOp(1      @9      @A      @I      @a??	*+R??i`?iRT????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?Nn??9?i*??#v????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?Nn??9?i?i???????Unknown
\'HostGreater"Greater(1      @9      @A      @I      @a?Nn??9?i?7?Ź????Unknown
?(HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?Nn??9?i???ۗ???Unknown
?)HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?Nn??9?iR??g?????Unknown
~*HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?Nn??9?i?9????Unknown
v+HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?Nn??9?i?n:
A????Unknown
`,HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?Nn??9?i?<X?b????Unknown
?-HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?Nn??9?iz
v??????Unknown
?.Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?Nn??9?iDؓ}?????Unknown
?/HostMatMul"-gradient_tape/sequential_10/dense_33/MatMul_1(1      @9      @A      @I      @a?Nn??9?i??Nȭ???Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a>?Ҳ??2?ie ??!????Unknown
X1HostEqual"Equal(1      @9      @A      @I      @a>?Ҳ??2?i?Z^{????Unknown
V2HostMean"Mean(1      @9      @A      @I      @a>?Ҳ??2?i?4eԴ???Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a>?Ҳ??2?ij?-????Unknown
|4HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a>?Ҳ??2?i?i??????Unknown
j5HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a>?Ҳ??2?iķ{?????Unknown
r6HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a>?Ҳ??2?io??9????Unknown
v7HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a>?Ҳ??2?i?xd5?????Unknown
?8HostReadVariableOp",sequential_10/dense_31/MatMul/ReadVariableOp(1      @9      @A      @I      @a>?Ҳ??2?i?:??????Unknown
?9HostReadVariableOp"-sequential_10/dense_32/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a>?Ҳ??2?it-?E????Unknown
?:HostReadVariableOp"-sequential_10/dense_33/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a>?Ҳ??2?iˇ?K?????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?Nn??)?i?nv40????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?Nn??)?i?U?????Unknown
V=HostCast"Cast(1       @9       @A       @I       @a?Nn??)?iz<?R????Unknown
X>HostCast"Cast_3(1       @9       @A       @I       @a?Nn??)?i_##??????Unknown
X?HostCast"Cast_5(1       @9       @A       @I       @a?Nn??)?iD
??s????Unknown
T@HostMul"Mul(1       @9       @A       @I       @a?Nn??)?i)?@?????Unknown
vAHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?Nn??)?i?ϧ?????Unknown
zBHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?Nn??)?i??^?&????Unknown
vCHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?Nn??)?iإ?x?????Unknown
?DHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?Nn??)?i??|aH????Unknown
}EHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?Nn??)?i?sJ?????Unknown
wFHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @a?Nn??)?i?Z?2j????Unknown
bGHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?Nn??)?ilA)?????Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?Nn??)?iQ(??????Unknown
xIHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?Nn??)?i6G?????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?Nn??)?i??ԭ????Unknown
?KHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?Nn??)?i ?d?>????Unknown
?LHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?Nn??)?i?????????Unknown
?MHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a?Nn??)?iʪ??`????Unknown
?NHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?Nn??)?i??w?????Unknown
?OHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?Nn??)?i?x?_?????Unknown
~PHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?Nn??)?iy_/H????Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?Nn??)?i^F?0?????Unknown
?RHostReluGrad"-gradient_tape/sequential_10/dense_31/ReluGrad(1       @9       @A       @I       @a?Nn??)?iC-M5????Unknown
?SHostReluGrad"-gradient_tape/sequential_10/dense_32/ReluGrad(1       @9       @A       @I       @a?Nn??)?i(??????Unknown
?THostReadVariableOp"-sequential_10/dense_31/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?Nn??)?i?j?V????Unknown
?UHostReadVariableOp",sequential_10/dense_33/MatMul/ReadVariableOp(1       @9       @A       @I       @a?Nn??)?i?????????Unknown
sVHostSigmoid"sequential_10/dense_33/Sigmoid(1       @9       @A       @I       @a?Nn??)?i?Ȉ?x????Unknown
XWHostCast"Cast_4(1      ??9      ??A      ??I      ??a?Nn???iI<?/A????Unknown
aXHostIdentity"Identity(1      ??9      ??A      ??I      ??a?Nn???i???	????Unknown?
?YHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a?Nn???i-#_?????Unknown
uZHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?Nn???i?????????Unknown
d[HostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?Nn???i
? c????Unknown
u\HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?Nn???i?}5u+????Unknown
y]HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?Nn???i??|??????Unknown
?^HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?Nn???igd?]?????Unknown
?_HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?Nn???i??҄????Unknown
?`HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?Nn???iKKSFM????Unknown
?aHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?Nn???i????????Unknown
?bHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?Nn???i/2?.?????Unknown
?cHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?Nn???i??)??????Unknown
?dHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?Nn???iqo????Unknown
?eHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?Nn???i????7????Unknown
?fHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?Nn???i?????????Unknown
igHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
YhHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?e
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?? x^??i?? x^???Unknown?
xHost_FusedMatMul"sequential_10/dense_33/BiasAdd(1      F@9      F@A      F@I      F@a?
???i??hn(p???Unknown
iHostWriteSummary"WriteSummary(1     ?A@9     ?A@A     ?A@I     ?A@a??? ?6??i2Zpf?I???Unknown?
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1      @@9      @@A      ?@I      ?@a??%?o??iR?	??
???Unknown
uHost_FusedMatMul"sequential_10/dense_31/Relu(1      =@9      =@A      =@I      =@a??;?W???iGe??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      ;@9      ;@A      ;@I      ;@a&9R%@???i??g???Unknown
?HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      5@9      5@A      5@I      5@a?H? ?S??iW?k?????Unknown
?HostMatMul"-gradient_tape/sequential_10/dense_32/MatMul_1(1      1@9      1@A      1@I      1@a??%?o??iG???cS???Unknown
?	HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      .@9      .@A      .@I      .@aG???cS??it?F?????Unknown
?
HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@aG???cS??i?6n?????Unknown
HostMatMul"+gradient_tape/sequential_10/dense_31/MatMul(1      *@9      *@A      *@I      *@apf?I47??ik????^???Unknown
dHostDataset"Iterator::Model(1      B@9      B@A      "@I      "@a??mܪ?{?i??N?֖???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?V?n{?x?i[?+??????Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a?V?n{?x?i??\????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a?V?n{?x?i?%??,???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1       @9       @A       @I       @a?V?n{?x?ibZ???]???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1       @9       @A       @I       @a?V?n{?x?i??Υ????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a?V?n{?x?i??}?h????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?V?n{?x?ii?Z?+????Unknown
HostMatMul"+gradient_tape/sequential_10/dense_32/MatMul(1       @9       @A       @I       @a?V?n{?x?i-8??$???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?? L?u?i.?9KyP???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a?? L?u?iFI;?|???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?? L?u?i^?<{?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a?? L?u?ive>????Unknown
HostMatMul"+gradient_tape/sequential_10/dense_33/MatMul(1      @9      @A      @I      @a?? L?u?i?????????Unknown
uHost_FusedMatMul"sequential_10/dense_32/Relu(1      @9      @A      @I      @a?? L?u?i??AC.*???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?? L?u?i?C۸U???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a????r?i@?h{???Unknown
?HostBiasAddGrad"8gradient_tape/sequential_10/dense_31/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????r?i?ގM]????Unknown
?HostBiasAddGrad"8gradient_tape/sequential_10/dense_32/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a????r?iDƴ??????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a^?@J?o?i0?`?????Unknown
v HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a^?@J?o?iHI;????Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a^?@J?o?i???"???Unknown
?"HostBiasAddGrad"8gradient_tape/sequential_10/dense_33/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a^?@J?o?i????B???Unknown
?#HostReadVariableOp",sequential_10/dense_32/MatMul/ReadVariableOp(1      @9      @A      @I      @a^?@J?o?i?
(?0a???Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?V?n{?h?i7??Ez???Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?V?n{?h?i????????Unknown
\&HostGreater"Greater(1      @9      @A      @I      @a?V?n{?h?i??s<ի???Unknown
?'HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?V?n{?h?i<tⷶ????Unknown
?(HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a?V?n{?h?i?Q3?????Unknown
~)HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a?V?n{?h?i꨿?y????Unknown
v*HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a?V?n{?h?iAC.*[???Unknown
`+HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?V?n{?h?i?ݜ?<(???Unknown
?,HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a?V?n{?h?i?w!A???Unknown
?-Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?V?n{?h?iFz??Y???Unknown
?.HostMatMul"-gradient_tape/sequential_10/dense_33/MatMul_1(1      @9      @A      @I      @a?V?n{?h?i????r???Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a????b?i^?{4?????Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a????b?i?Q3????Unknown
V1HostMean"Mean(1      @9      @A      @I      @a????b?i???mܪ???Unknown
s2HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a????b?i?{4??????Unknown
|3HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a????b?iboǦ.????Unknown
j4HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a????b?i#cZ??????Unknown
r5HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a????b?i?V?߀????Unknown
v6HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a????b?i?J??)???Unknown
?7HostReadVariableOp",sequential_10/dense_31/MatMul/ReadVariableOp(1      @9      @A      @I      @a????b?if>????Unknown
?8HostReadVariableOp"-sequential_10/dense_32/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????b?i'2?5|-???Unknown
?9HostReadVariableOp"-sequential_10/dense_33/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a????b?i?%9R%@???Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?V?n{?X?is??L???Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?V?n{?X?i>???Y???Unknown
V<HostCast"Cast(1       @9       @A       @I       @a?V?n{?X?ii_?we???Unknown
X=HostCast"Cast_3(1       @9       @A       @I       @a?V?n{?X?i?ZI?q???Unknown
X>HostCast"Cast_5(1       @9       @A       @I       @a?V?n{?X?i???Y~???Unknown
T?HostMul"Mul(1       @9       @A       @I       @a?V?n{?X?i????Ɋ???Unknown
v@HostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?V?n{?X?iB<?:????Unknown
zAHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1       @9       @A       @I       @a?V?n{?X?i@????????Unknown
vBHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?V?n{?X?ikܪ?????Unknown
?CHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1       @9       @A       @I       @a?V?n{?X?i?)b??????Unknown
}DHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?V?n{?X?i?vy?????Unknown
wEHostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @a?V?n{?X?i???6n????Unknown
bFHostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a?V?n{?X?i???????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?V?n{?X?iB^??O????Unknown
xHHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a?V?n{?X?im??o?????Unknown
~IHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?V?n{?X?i???-1???Unknown
?JHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1       @9       @A       @I       @a?V?n{?X?i?Ee?????Unknown
?KHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?V?n{?X?i??? ???Unknown
?LHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1       @9       @A       @I       @a?V?n{?X?i??f?,???Unknown
?MHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1       @9       @A       @I       @a?V?n{?X?iD-?$?8???Unknown
?NHost	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1       @9       @A       @I       @a?V?n{?X?iozB?dE???Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?V?n{?X?i?????Q???Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?V?n{?X?i??]F^???Unknown
?QHostReluGrad"-gradient_tape/sequential_10/dense_31/ReluGrad(1       @9       @A       @I       @a?V?n{?X?i?ah?j???Unknown
?RHostReluGrad"-gradient_tape/sequential_10/dense_32/ReluGrad(1       @9       @A       @I       @a?V?n{?X?i??'w???Unknown
?SHostReadVariableOp"-sequential_10/dense_31/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?V?n{?X?iF?֖?????Unknown
?THostReadVariableOp",sequential_10/dense_33/MatMul/ReadVariableOp(1       @9       @A       @I       @a?V?n{?X?iqI?T	????Unknown
sUHostSigmoid"sequential_10/dense_33/Sigmoid(1       @9       @A       @I       @a?V?n{?X?i??Ez????Unknown
XVHostCast"Cast_4(1      ??9      ??A      ??I      ??a?V?n{?H?i2=!q?????Unknown
aWHostIdentity"Identity(1      ??9      ??A      ??I      ??a?V?n{?H?i?????????Unknown?
?XHostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1      ??9      ??A      ??I      ??a?V?n{?H?i^??.#????Unknown
uYHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?V?n{?H?i?0??[????Unknown
dZHostAddN"SGD/gradients/AddN(1      ??9      ??A      ??I      ??a?V?n{?H?i?׏쓻???Unknown
u[HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?V?n{?H?i ~kK?????Unknown
y\HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?V?n{?H?i?$G?????Unknown
?]HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1      ??9      ??A      ??I      ??a?V?n{?H?iL?"	=????Unknown
?^HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?V?n{?H?i?q?gu????Unknown
?_HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?V?n{?H?ix?ƭ????Unknown
?`Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?V?n{?H?i??%?????Unknown
?aHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?V?n{?H?i?e??????Unknown
?bHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?V?n{?H?i:m?V????Unknown
?cHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?V?n{?H?iвHB?????Unknown
?dHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?V?n{?H?ifY$??????Unknown
?eHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?V?n{?H?i?????????Unknown
ifHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
YgHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown2CPU