"?p
BHostIDLE"IDLE1     ??@A     ??@a???m??i???m???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a?wy0l??i?$ɚ????Unknown?
`HostGatherV2"
GatherV2_1(1      G@9      G@A      G@I      G@aE??o?s?i?|????Unknown
^HostGatherV2"GatherV2(1     ?E@9     ?E@A     ?E@I     ?E@aL????tr?i<?B'???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      >@9      >@A      >@I      >@a|O???i?iC_W?@???Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@a?SzPx-g?i?٧y?W???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@a8:?Qf?iYb?/An???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      5@9      5@A      5@I      5@a?????b?iE3?H????Unknown
?	HostReadVariableOp"-sequential_62/dense_195/MatMul/ReadVariableOp(1      5@9      5@A      5@I      5@a?????b?i1|O????Unknown
?
HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      4@9      4@A      4@I      4@aSZ߶)+a?i??21z????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      0@I      0@a*2?Bx[?i???R6????Unknown
vHost_FusedMatMul"sequential_62/dense_192/Relu(1      0@9      0@A      0@I      0@a*2?Bx[?i??s?????Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a*2?Bx[?i?.P??????Unknown
rHostAdd"!binary_crossentropy/logistic_loss(1      (@9      (@A      (@I      (@a???2?T?i??W??????Unknown
yHost_FusedMatMul"sequential_62/dense_195/BiasAdd(1      (@9      (@A      (@I      (@a???2?T?ij?^?H????Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a?|????R?i?5P?????Unknown
?HostMatMul",gradient_tape/sequential_62/dense_194/MatMul(1      "@9      "@A      "@I      "@abo?K?N?iĳ?s????Unknown
?HostMatMul".gradient_tape/sequential_62/dense_194/MatMul_1(1      "@9      "@A      "@I      "@abo?K?N?i?1??-????Unknown
dHostDataset"Iterator::Model(1      C@9      C@A       @I       @a*2?BxK?ik??????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a*2?BxK?i?J:?????Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a*2?BxK?i????????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a*2?BxK?id?????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a*2?BxK?i??H????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a*2?BxK?i"}?'b#???Unknown
?HostMatMul",gradient_tape/sequential_62/dense_193/MatMul(1       @9       @A       @I       @a*2?BxK?i?	?8@*???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a??kf:	H?i??A?B0???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a??kf:	H?i????D6???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @a??kf:	H?i??t$G<???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a???2?D?i ???mA???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a???2?D?ih-|=?F???Unknown
?HostMatMul",gradient_tape/sequential_62/dense_192/MatMul(1      @9      @A      @I      @a???2?D?i???ɺK???Unknown
? HostBiasAddGrad"9gradient_tape/sequential_62/dense_193/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???2?D?i8??V?P???Unknown
?!HostMatMul".gradient_tape/sequential_62/dense_193/MatMul_1(1      @9      @A      @I      @a???2?D?i?)?V???Unknown
?"HostMatMul",gradient_tape/sequential_62/dense_195/MatMul(1      @9      @A      @I      @a???2?D?iӊo.[???Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aSZ߶)+A?iߊ?9y_???Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aSZ߶)+A?i?Bf?c???Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aSZ߶)+A?i????h???Unknown
V&HostSum"Sum_2(1      @9      @A      @I      @aSZ߶)+A?id?A?Yl???Unknown
?'HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aSZ߶)+A?i;j?c?p???Unknown
~(HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @aSZ߶)+A?i".?t???Unknown
?)HostBiasAddGrad"9gradient_tape/sequential_62/dense_192/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aSZ߶)+A?i?ي?9y???Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_62/dense_194/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aSZ߶)+A?i???}???Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_62/dense_195/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aSZ߶)+A?i?If?ρ???Unknown
?,HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a*2?Bx;?i???>????Unknown
~-HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a*2?Bx;?i!???????Unknown
v.HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a*2?Bx;?if?m?????Unknown
?/HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a*2?Bx;?i?bŮ?????Unknown
?0Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a*2?Bx;?i?(??????Unknown
?1HostMatMul".gradient_tape/sequential_62/dense_195/MatMul_1(1      @9      @A      @I      @a*2?Bx;?i5?t?i????Unknown
?2HostReadVariableOp".sequential_62/dense_192/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a*2?Bx;?iz???ؙ???Unknown
v3Host_FusedMatMul"sequential_62/dense_193/Relu(1      @9      @A      @I      @a*2?Bx;?i?{$?G????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a???2?4?isPf۟???Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a???2?4?i'%?\n????Unknown
X6HostCast"Cast_3(1      @9      @A      @I      @a???2?4?i????????Unknown
\7HostGreater"Greater(1      @9      @A      @I      @a???2?4?i??+锧???Unknown
V8HostMean"Mean(1      @9      @A      @I      @a???2?4?iC?m/(????Unknown
?9HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a???2?4?i?w?u?????Unknown
v:HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a???2?4?i?L??N????Unknown
v;HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???2?4?i_!3?????Unknown
?<HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a???2?4?i?tHu????Unknown
b=HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a???2?4?i?ʶ?????Unknown
?>HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???2?4?i{??ԛ????Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a???2?4?i/t:/????Unknown
v@Host_FusedMatMul"sequential_62/dense_194/Relu(1      @9      @A      @I      @a???2?4?i?H|a¾???Unknown
?AHostReadVariableOp".sequential_62/dense_195/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a???2?4?i???U????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a*2?Bx+?i? ?+????Unknown
VCHostCast"Cast(1       @9       @A       @I       @a*2?Bx+?i????????Unknown
XDHostCast"Cast_5(1       @9       @A       @I       @a*2?Bx+?i ?A4|????Unknown
XEHostEqual"Equal(1       @9       @A       @I       @a*2?Bx+?i#?m?3????Unknown
uFHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a*2?Bx+?iF??<?????Unknown
|GHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a*2?Bx+?iip???????Unknown
dHHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a*2?Bx+?i?S?DZ????Unknown
vIHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a*2?Bx+?i?6?????Unknown
vJHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a*2?Bx+?i?IM?????Unknown
}KHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a*2?Bx+?i??tр????Unknown
`LHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a*2?Bx+?i??U8????Unknown
uMHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a*2?Bx+?i;????????Unknown
wNHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a*2?Bx+?i^??]?????Unknown
?OHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a*2?Bx+?i??$?^????Unknown
xPHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a*2?Bx+?i?lPf????Unknown
?QHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a*2?Bx+?i?O|??????Unknown
?RHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a*2?Bx+?i?2?n?????Unknown
?SHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a*2?Bx+?i??<????Unknown
?THostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a*2?Bx+?i0??v?????Unknown
~UHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a*2?Bx+?iS?+??????Unknown
?VHostReluGrad".gradient_tape/sequential_62/dense_192/ReluGrad(1       @9       @A       @I       @a*2?Bx+?iv?Wc????Unknown
?WHostReluGrad".gradient_tape/sequential_62/dense_193/ReluGrad(1       @9       @A       @I       @a*2?Bx+?i???????Unknown
?XHostReluGrad".gradient_tape/sequential_62/dense_194/ReluGrad(1       @9       @A       @I       @a*2?Bx+?i?????????Unknown
?YHostReadVariableOp".sequential_62/dense_193/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a*2?Bx+?i?h??????Unknown
?ZHostReadVariableOp".sequential_62/dense_194/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a*2?Bx+?iL?A????Unknown
t[HostSigmoid"sequential_62/dense_195/Sigmoid(1       @9       @A       @I       @a*2?Bx+?i%/3?????Unknown
t\HostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a*2?Bx?i? I??????Unknown
X]HostCast"Cast_4(1      ??9      ??A      ??I      ??a*2?Bx?iG_??????Unknown
a^HostIdentity"Identity(1      ??9      ??A      ??I      ??a*2?Bx?i?uZ?????Unknown?
T_HostMul"Mul(1      ??9      ??A      ??I      ??a*2?Bx?ii??h????Unknown
s`HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a*2?Bx?i????C????Unknown
jaHostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a*2?Bx?i?ض?????Unknown
wbHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a*2?Bx?i??b?????Unknown
ycHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a*2?Bx?i???$?????Unknown
?dHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a*2?Bx?i>????????Unknown
?eHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a*2?Bx?iϞ??????Unknown
?fHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a*2?Bx?i`?$kj????Unknown
?gHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a*2?Bx?i??:-F????Unknown
?hHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a*2?Bx?i?sP?!????Unknown
?iHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a*2?Bx?ief??????Unknown
?jHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a*2?Bx?i?V|s?????Unknown
?kHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a*2?Bx?i5H?5?????Unknown
?lHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a*2?Bx?i?9???????Unknown
?mHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a*2?Bx?iW+??l????Unknown
?nHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a*2?Bx?i??{H????Unknown
?oHostReadVariableOp"-sequential_62/dense_192/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a*2?Bx?iy?=$????Unknown
?pHostReadVariableOp"-sequential_62/dense_193/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a*2?Bx?i     ???Unknown
?qHostReadVariableOp"-sequential_62/dense_194/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a*2?Bx?i??
?m ???Unknown*?o
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a??????i???????Unknown?
`HostGatherV2"
GatherV2_1(1      G@9      G@A      G@I      G@aAA??i? ? ???Unknown
^HostGatherV2"GatherV2(1     ?E@9     ?E@A     ?E@I     ?E@a!!??i?0?0???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      >@9      >@A      >@I      >@ah?h???i4<?3<????Unknown
iHostWriteSummary"WriteSummary(1      ;@9      ;@A      ;@I      ;@aDADA??iUFeTFe???Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      :@9      :@A      :@I      :@a8?8???iPP???Unknown
zHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      5@9      5@A      5@I      5@a ??????i?W?W???Unknown
?HostReadVariableOp"-sequential_62/dense_195/MatMul/ReadVariableOp(1      5@9      5@A      5@I      5@a ??????i?_??_????Unknown
?	HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      4@9      4@A      4@I      4@a????iVguVgu???Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      0@I      0@a????iVm?Vm????Unknown
vHost_FusedMatMul"sequential_62/dense_192/Relu(1      0@9      0@A      0@I      0@a????iVs5Ws5???Unknown
gHostStridedSlice"strided_slice(1      0@9      0@A      0@I      0@a????iVy?Wy????Unknown
rHostAdd"!binary_crossentropy/logistic_loss(1      (@9      (@A      (@I      (@a  ??i?}??}????Unknown
yHost_FusedMatMul"sequential_62/dense_195/BiasAdd(1      (@9      (@A      (@I      (@a  ??iV?%X?%???Unknown
lHostIteratorGetNext"IteratorGetNext(1      &@9      &@A      &@I      &@a????iv?gx?g???Unknown
?HostMatMul",gradient_tape/sequential_62/dense_194/MatMul(1      "@9      "@A      "@I      "@a??{?i։?؉????Unknown
?HostMatMul".gradient_tape/sequential_62/dense_194/MatMul_1(1      "@9      "@A      "@I      "@a??{?i6??8?????Unknown
dHostDataset"Iterator::Model(1      C@9      C@A       @I       @a??x?i6?9????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a??x?i6?39?3???Unknown?
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1       @9       @A       @I       @a??x?i6?c9?c???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_7/ResourceApplyGradientDescent(1       @9       @A       @I       @a??x?i6??9?????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1       @9       @A       @I       @a??x?i6??9?????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a??x?i6??9?????Unknown
?HostMatMul",gradient_tape/sequential_62/dense_193/MatMul(1       @9       @A       @I       @a??x?i6?#:?#???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aPPu?i֤MڤM???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPu?iv?wz?w???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_6/ResourceApplyGradientDescent(1      @9      @A      @I      @aPPu?i???????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      @9      @A      @I      @a  r?iV??Z?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a  r?i??隮????Unknown
?HostMatMul",gradient_tape/sequential_62/dense_192/MatMul(1      @9      @A      @I      @a  r?iְ۰???Unknown
?HostBiasAddGrad"9gradient_tape/sequential_62/dense_193/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a  r?i?1?1???Unknown
? HostMatMul".gradient_tape/sequential_62/dense_193/MatMul_1(1      @9      @A      @I      @a  r?iV?U[?U???Unknown
?!HostMatMul",gradient_tape/sequential_62/dense_195/MatMul(1      @9      @A      @I      @a  r?i??y??y???Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??n?iv??{?????Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a??n?iV??[?????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a??n?i6??;?????Unknown
V%HostSum"Sum_2(1      @9      @A      @I      @a??n?i???????Unknown
?&HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??n?i???????Unknown
~'HostMaximum")gradient_tape/binary_crossentropy/Maximum(1      @9      @A      @I      @a??n?i??-??-???Unknown
?(HostBiasAddGrad"9gradient_tape/sequential_62/dense_192/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??n?i??K??K???Unknown
?)HostBiasAddGrad"9gradient_tape/sequential_62/dense_194/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??n?i??i??i???Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_62/dense_195/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??n?ivȇ|ȇ???Unknown
?+HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a??h?i?ɟ?ɟ???Unknown
~,HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a??h?iv˷|˷???Unknown
v-HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a??h?i?????????Unknown
?.HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a??h?iv??|?????Unknown
?/Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a??h?i?????????Unknown
?0HostMatMul".gradient_tape/sequential_62/dense_195/MatMul_1(1      @9      @A      @I      @a??h?iv?}????Unknown
?1HostReadVariableOp".sequential_62/dense_192/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a??h?i??/??/???Unknown
v2Host_FusedMatMul"sequential_62/dense_193/Relu(1      @9      @A      @I      @a??h?iv?G}?G???Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a  b?i??Y??Y???Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a  b?i??k??k???Unknown
X5HostCast"Cast_3(1      @9      @A      @I      @a  b?i??}??}???Unknown
\6HostGreater"Greater(1      @9      @A      @I      @a  b?i?؏?؏???Unknown
V7HostMean"Mean(1      @9      @A      @I      @a  b?iڡڡ???Unknown
?8HostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1      @9      @A      @I      @a  b?i6۳=۳???Unknown
v9HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a  b?iV??]?????Unknown
v:HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a  b?iv??}?????Unknown
?;HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a  b?i?????????Unknown
b<HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a  b?i?????????Unknown
?=HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a  b?i???????Unknown
?>Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a  b?i???????Unknown
v?Host_FusedMatMul"sequential_62/dense_194/Relu(1      @9      @A      @I      @a  b?i?1?1???Unknown
?@HostReadVariableOp".sequential_62/dense_195/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a  b?i6?C>?C???Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a??X?i??O??O???Unknown
VBHostCast"Cast(1       @9       @A       @I       @a??X?i??[??[???Unknown
XCHostCast"Cast_5(1       @9       @A       @I       @a??X?iv?g~?g???Unknown
XDHostEqual"Equal(1       @9       @A       @I       @a??X?i6?s>?s???Unknown
uEHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1       @9       @A       @I       @a??X?i???????Unknown
|FHostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a??X?i?苾?????Unknown
dGHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a??X?iv??~?????Unknown
vHHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??X?i6??>?????Unknown
vIHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??X?i?????????Unknown
}JHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??X?i?뻾?????Unknown
`KHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??X?iv??~?????Unknown
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a??X?i6??>?????Unknown
wMHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a??X?i?????????Unknown
?NHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??X?i?????????Unknown
xOHostCast"&gradient_tape/binary_crossentropy/Cast(1       @9       @A       @I       @a??X?iv??~?????Unknown
?PHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1       @9       @A       @I       @a??X?i6??????Unknown
?QHostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1       @9       @A       @I       @a??X?i???????Unknown
?RHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a??X?i???????Unknown
?SHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1       @9       @A       @I       @a??X?iv?'?'???Unknown
~THostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??X?i6?3??3???Unknown
?UHostReluGrad".gradient_tape/sequential_62/dense_192/ReluGrad(1       @9       @A       @I       @a??X?i?????????Unknown
?VHostReluGrad".gradient_tape/sequential_62/dense_193/ReluGrad(1       @9       @A       @I       @a??X?i??K??K???Unknown
?WHostReluGrad".gradient_tape/sequential_62/dense_194/ReluGrad(1       @9       @A       @I       @a??X?iv?W?W???Unknown
?XHostReadVariableOp".sequential_62/dense_193/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??X?i6?c??c???Unknown
?YHostReadVariableOp".sequential_62/dense_194/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??X?i??o??o???Unknown
tZHostSigmoid"sequential_62/dense_195/Sigmoid(1       @9       @A       @I       @a??X?i??{??{???Unknown
t[HostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a??H?i???????Unknown
X\HostCast"Cast_4(1      ??9      ??A      ??I      ??a??H?iv???????Unknown
a]HostIdentity"Identity(1      ??9      ??A      ??I      ??a??H?i?????????Unknown?
T^HostMul"Mul(1      ??9      ??A      ??I      ??a??H?i6????????Unknown
s_HostReadVariableOp"SGD/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
j`HostMean"binary_crossentropy/Mean(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??H?iV??_?????Unknown
ybHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
?cHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??H?i???????Unknown
?dHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a??H?iv???????Unknown
?eHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a??H?i6????????Unknown
?gHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??H?i??ɟ?????Unknown
?hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??H?iV??_?????Unknown
?jHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a??H?i??ۿ?????Unknown
?kHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??H?i???????Unknown
?lHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??H?iv???????Unknown
?mHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
?nHostReadVariableOp"-sequential_62/dense_192/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??H?i6????????Unknown
?oHostReadVariableOp"-sequential_62/dense_193/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??H?i?????????Unknown
?pHostReadVariableOp"-sequential_62/dense_194/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??H?i?????????Unknown2CPU