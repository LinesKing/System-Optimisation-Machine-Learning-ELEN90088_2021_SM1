"?e
BHostIDLE"IDLE1     z?@A     z?@a
#?f??i
#?f???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?P@9     ?P@A     ?P@I     ?P@a??K2{???iـl@6 ???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      G@9      G@A      D@I      D@a`kZ?ך??i??ퟡz???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@a7F?̈́??i??ִ????Unknown?
qHost_FusedMatMul"sequential/dense_1/Relu(1      =@9      =@A      =@I      =@a؍Avc??i
	0?B???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      :@I      :@a??up?b}?i!?z=???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      :@9      :@A      :@I      :@a??up?b}?i8??D?w???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      5@9      5@A      5@I      5@a??^x/?w?i???F????Unknown
?	HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      4@9      4@A      4@I      4@a`kZ?ךv?i?Q?S|????Unknown
?
Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      .@9      .@A      .@I      .@a??C?!?p?i??s?d????Unknown
XHostEqual"Equal(1      ,@9      ,@A      ,@I      ,@aS?~???o?i'X,
???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      ,@9      ,@A      ,@I      ,@aS?~???o?i?ִ??5???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@aS?~???o?iSUUUUU???Unknown
dHostDataset"Iterator::Model(1     ?S@9     ?S@A      (@I      (@aـl@6 k?i????up???Unknown
VHostCast"Cast(1      &@9      &@A      &@I      &@avc??h?iJ%?S????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@avc??h?i????0????Unknown?
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a?`Q?(Xd?i!?f????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      "@9      "@A      "@I      "@a?`Q?(Xd?i?+??????Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?UH?yb?i?s?d?????Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @aS?~???_?i#??.?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @aS?~???_?in?7??????Unknown
VHostSum"Sum_2(1      @9      @A      @I      @aS?~???_?i?1??n???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @aS?~???_?iq؍A???Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aS?~???_?iO?(X,???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aـl@6 [?i??Hs?9???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @aـl@6 [?i?i?4G???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aـl@6 [?iS???T???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aـl@6 [?iO???Tb???Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @aـl@6 [?i?????o???Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aـl@6 [?i????t}???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aـl@6 [?i,
????Unknown
v HostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @a`kZ?ךV?iEY??R????Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a`kZ?ךV?i{??ퟡ???Unknown
?"HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a`kZ?ךV?i???Y?????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?UH?yR?i?ך?????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?UH?yR?i?Z?????Unknown
?%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?UH?yR?i2 ?????Unknown
r&HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?UH?yR?i]D?L????Unknown
z'HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?UH?yR?i?h?	#????Unknown
v(HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?UH?yR?i??[?-????Unknown
v)HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?UH?yR?iް?8????Unknown
b*HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?UH?yR?i	???C????Unknown
?+Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?UH?yR?i4???M????Unknown
?,HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?UH?yR?i_\?X???Unknown
?-HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?UH?yR?i?Avc???Unknown
?.HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?UH?yR?i?e?2n???Unknown
?/HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?UH?yR?i????x"???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aـl@6 K?i ?,?@)???Unknown
?1HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?H@9     ?H@A      @I      @aـl@6 K?i ??
	0???Unknown
e2Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aـl@6 K?i@?L?6???Unknown?
V3HostMean"Mean(1      @9      @A      @I      @aـl@6 K?i`??%?=???Unknown
s4HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aـl@6 K?i?m3aD???Unknown
j5HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @aـl@6 K?i?,?@)K???Unknown
~6HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @aـl@6 K?i?G?N?Q???Unknown
?7HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aـl@6 K?i?b\?X???Unknown
?8HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @aـl@6 K?i ~?i?_???Unknown
?9HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @aـl@6 K?i ?=wIf???Unknown
?:HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @aـl@6 K?i@?̈́m???Unknown
o;HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @aـl@6 K?i`?]??s???Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?UH?yB?iu???^x???Unknown
X=HostCast"Cast_3(1       @9       @A       @I       @a?UH?yB?i??O?|???Unknown
X>HostCast"Cast_5(1       @9       @A       @I       @a?UH?yB?i?~?i????Unknown
\?HostGreater"Greater(1       @9       @A       @I       @a?UH?yB?i???????Unknown
|@HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?UH?yB?i?)>jt????Unknown
dAHostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?UH?yB?i?;???????Unknown
vBHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?UH?yB?i?M?&????Unknown
?CHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?UH?yB?i`^?????Unknown
vDHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?UH?yB?ir?㉜???Unknown
`EHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?UH?yB?i2?B????Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?UH?yB?iG?~??????Unknown
wGHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?UH?yB?i\???????Unknown
?HHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?UH?yB?iq?>]?????Unknown
~IHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?UH?yB?i?̞?$????Unknown
?JHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?UH?yB?i????????Unknown
~KHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?UH?yB?i??^x/????Unknown
}LHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?UH?yB?i??ִ????Unknown
MHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?UH?yB?i?5:????Unknown
}NHostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @a?UH?yB?i?&??????Unknown
?OHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?UH?yB?i9??D????Unknown
tPHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?UH?y2?iB??????Unknown
vQHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?UH?y2?iK?P?????Unknown
XRHostCast"Cast_4(1      ??9      ??A      ??I      ??a?UH?y2?i%To?????Unknown
TSHostMul"Mul(1      ??9      ??A      ??I      ??a?UH?y2?i0]??O????Unknown
uTHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?UH?y2?i;f?]?????Unknown
?UHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?UH?y2?iFo??????Unknown
}VHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?UH?y2?iQx/?????Unknown
wWHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?UH?y2?i\?_kZ????Unknown
xXHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?UH?y2?ig???????Unknown
?YHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?UH?y2?ir????????Unknown
?ZHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?UH?y2?i}??x"????Unknown
?[Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?UH?y2?i??(e????Unknown
?\HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?UH?y2?i??Oק????Unknown
?]HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?UH?y2?i????????Unknown
?^HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?UH?y2?i???5-????Unknown
?_HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?UH?y2?i????o????Unknown
?`HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?UH?y2?i????????Unknown
?aHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?UH?y2?i???C?????Unknown
?bHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?UH?y2?i??o?7????Unknown
?cHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?UH?y2?i?ퟡz????Unknown
?dHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?UH?y2?i???P?????Unknown
?eHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?UH?y2?i?????????Unknown
4fHostIdentity"Identity(i?????????Unknown?
igHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
LhHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
[iHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown*?d
sHostDataset"Iterator::Model::ParallelMapV2(1     ?P@9     ?P@A     ?P@I     ?P@a?҆?????i?҆??????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      G@9      G@A      D@I      D@aHv&?????i???+B ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      ?@9      ?@A      ?@I      ?@a#b?/???i?%/*N????Unknown?
qHost_FusedMatMul"sequential/dense_1/Relu(1      =@9      =@A      =@I      =@a?^????i??Y5=D???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      :@9      :@A      :@I      :@a?f?r???iz?<?k????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      :@9      :@A      :@I      :@a?f?r???iL?ٙ????Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      5@9      5@A      5@I      5@a???$?^??il??}p????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      4@9      4@A      4@I      4@aHv&?????i5??2?????Unknown
?	Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      .@9      .@A      .@I      .@al??}pИ?iL?:?n???Unknown
X
HostEqual"Equal(1      ,@9      ,@A      ,@I      ,@a?i??(??i? Se????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1      ,@9      ,@A      ,@I      ,@a?i??(??i?<?k?S???Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      ,@9      ,@A      ,@I      ,@a?i??(??i????Ac???Unknown
dHostDataset"Iterator::Model(1     ?S@9     ?S@A      (@I      (@a#?Ǘ?ٓ?i?$?^???Unknown
VHostCast"Cast(1      &@9      &@A      &@I      &@a?w?t2??i?ݸ?????Unknown
iHostWriteSummary"WriteSummary(1      &@9      &@A      &@I      &@a?w?t2??i????8%???Unknown?
lHostIteratorGetNext"IteratorGetNext(1      "@9      "@A      "@I      "@a???c?ƍ?iEK^T????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1      "@9      "@A      "@I      "@a???c?ƍ?i???p???Unknown
?HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1       @9       @A       @I       @a?V
u?w??i???O}???Unknown
^HostGatherV2"GatherV2(1      @9      @A      @I      @a?i??(??i$?Ǘ?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a?i??(??iTe?]?6???Unknown
VHostSum"Sum_2(1      @9      @A      @I      @a?i??(??i?	?#;????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      @9      @A      @I      @a?i??(??i????????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?i??(??i?Q.??L???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a#?Ǘ?ك?i?p?~?????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a#?Ǘ?ك?i???LR????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a#?Ǘ?ك?i??K?:???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a#?Ǘ?ك?i?ͪ?!????Unknown
?HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a#?Ǘ?ك?i??	??????Unknown
?HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a#?Ǘ?ك?ii??(???Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a#?Ǘ?ك?i+?TYx???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      @9      @A      @I      @aHv&?????i??l+?????Unknown
? HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aHv&?????i?^?????Unknown
?!HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @aHv&?????i?????>???Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?V
u?wz?i@???s???Unknown
?#HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      @9      @A      @I      @a?V
u?wz?i?!???????Unknown
?$HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a?V
u?wz?i?6tu?????Unknown
r%HostAdd"!binary_crossentropy/logistic_loss(1      @9      @A      @I      @a?V
u?wz?iJK^T????Unknown
z&HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?V
u?wz?i?_H3?G???Unknown
v'HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?V
u?wz?i?t2||???Unknown
v(HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?V
u?wz?iT??k????Unknown
b)HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?V
u?wz?i??[????Unknown
?*Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1      @9      @A      @I      @a?V
u?wz?i????K???Unknown
?+HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?V
u?wz?i^?ڍ;P???Unknown
?,HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?V
u?wz?i??l+????Unknown
?-HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?V
u?wz?i???K????Unknown
?.HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?V
u?wz?ih?*????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a#?Ǘ??s?i???????Unknown
?0HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?H@9     ?H@A      @I      @a#?Ǘ??s?il$??r>???Unknown
e1Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a#?Ǘ??s?i??'?&f???Unknown?
V2HostMean"Mean(1      @9      @A      @I      @a#?Ǘ??s?ipCW?ڍ???Unknown
s3HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a#?Ǘ??s?i?҆??????Unknown
j4HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a#?Ǘ??s?itb??B????Unknown
~5HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      @9      @A      @I      @a#?Ǘ??s?i???|????Unknown
?6HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a#?Ǘ??s?ix?d?,???Unknown
?7HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      @9      @A      @I      @a#?Ǘ??s?i?EK^T???Unknown
?8HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a#?Ǘ??s?i|?t2|???Unknown
?9HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      @9      @A      @I      @a#?Ǘ??s?i?/?ƣ???Unknown
o:HostSigmoid"sequential/dense_2/Sigmoid(1      @9      @A      @I      @a#?Ǘ??s?i??? z????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a?V
u?wj?i??H??????Unknown
X<HostCast"Cast_3(1       @9       @A       @I       @a?V
u?wj?i.Խ?i ???Unknown
X=HostCast"Cast_5(1       @9       @A       @I       @a?V
u?wj?i??2?????Unknown
\>HostGreater"Greater(1       @9       @A       @I       @a?V
u?wj?i?觾Y5???Unknown
|?HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1       @9       @A       @I       @a?V
u?wj?i3???O???Unknown
d@HostAddN"SGD/gradients/AddN(1       @9       @A       @I       @a?V
u?wj?i????Ij???Unknown
vAHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?V
u?wj?i???????Unknown
?BHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?V
u?wj?i8||9????Unknown
vCHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?V
u?wj?i??k?????Unknown
`DHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?V
u?wj?i?&f[)????Unknown
uEHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?V
u?wj?i=1?J?????Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1       @9       @A       @I       @a?V
u?wj?i?;P:	???Unknown
?GHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?V
u?wj?i?E?)?#???Unknown
~HHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?V
u?wj?iBP:	>???Unknown
?IHostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1       @9       @A       @I       @a?V
u?wj?i?Z??X???Unknown
~JHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?V
u?wj?i?d$??r???Unknown
}KHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1       @9       @A       @I       @a?V
u?wj?iGo??p????Unknown
LHostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1       @9       @A       @I       @a?V
u?wj?i?y??????Unknown
}MHostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1       @9       @A       @I       @a?V
u?wj?i????`????Unknown
?NHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?V
u?wj?iL????????Unknown
tOHostAssignAddVariableOp"AssignAddVariableOp(1      ??9      ??A      ??I      ??a?V
u?wZ?iw??????Unknown
vPHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?V
u?wZ?i??m?P????Unknown
XQHostCast"Cast_4(1      ??9      ??A      ??I      ??a?V
u?wZ?i?(?????Unknown
TRHostMul"Mul(1      ??9      ??A      ??I      ??a?V
u?wZ?i????????Unknown
uSHostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?V
u?wZ?i#(?????Unknown
?THostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?V
u?wZ?iN?W?@,???Unknown
}UHostDivNoNan"'binary_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?V
u?wZ?iy2||9???Unknown
wVHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?V
u?wZ?i???s?F???Unknown
xWHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?V
u?wZ?i?<?k?S???Unknown
?XHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?V
u?wZ?i??Ac0a???Unknown
?YHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1      ??9      ??A      ??I      ??a?V
u?wZ?i%G?Zln???Unknown
?ZHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?V
u?wZ?iP̶R?{???Unknown
?[HostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1      ??9      ??A      ??I      ??a?V
u?wZ?i{QqJ?????Unknown
?\HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?V
u?wZ?i??+B ????Unknown
?]HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?V
u?wZ?i?[?9\????Unknown
?^HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?V
u?wZ?i???1?????Unknown
?_HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1      ??9      ??A      ??I      ??a?V
u?wZ?i'f[)Խ???Unknown
?`HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      ??9      ??A      ??I      ??a?V
u?wZ?iR?!????Unknown
?aHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?V
u?wZ?i}p?L????Unknown
?bHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?V
u?wZ?i????????Unknown
?cHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1      ??9      ??A      ??I      ??a?V
u?wZ?i?zE?????Unknown
?dHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?V
u?wZ?i?????????Unknown
4eHostIdentity"Identity(i?????????Unknown?
ifHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?????????Unknown
LgHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(i?????????Unknown
[hHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown2CPU