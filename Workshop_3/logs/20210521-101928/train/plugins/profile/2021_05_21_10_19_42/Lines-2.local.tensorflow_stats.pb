"?j
BHostIDLE"IDLE1    ?*?@A    ?*?@a ??]????i ??]?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@a??Х?O??i????5???Unknown?
~HostMatMul"*gradient_tape/sequential_4/dense_15/MatMul(1     ?D@9     ?D@A     ?D@I     ?D@a3???cvm?i??v?R???Unknown
?HostBiasAddGrad"7gradient_tape/sequential_4/dense_15/BiasAdd/BiasAddGrad(1      C@9      C@A      C@I      C@aO贁Nk?i?s??Mn???Unknown
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a?	???f?i?εL????Unknown?
tHost_FusedMatMul"sequential_4/dense_13/Relu(1      =@9      =@A      =@I      =@a??????d?i<?#????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@ap?}?a?i[ċ?????Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      7@9      7@A      7@I      7@a?n???`?i?k(??????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      2@9      2@A      2@I      2@a?j?ӕ?Y?i?R??????Unknown
?
HostReadVariableOp",sequential_4/dense_13/BiasAdd/ReadVariableOp(1      0@9      0@A      0@I      0@a?	???V?i? Y????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      .@I      .@a?X?0ҎU?i0a3??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@aU?gO?T?i[5?????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      *@9      *@A      *@I      *@a??Mn??R?i <??>????Unknown
oHostReadVariableOp"Adam/ReadVariableOp(1      *@9      *@A      *@I      *@a??Mn??R?i?b?/?????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      *@9      *@A      *@I      *@a??Mn??R?i?? ?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a$G4??Q?i$G4????Unknown
YHostPow"Adam/Pow(1      (@9      (@A      (@I      (@a$G4??Q?i@???,???Unknown
dHostDataset"Iterator::Model(1     ?B@9     ?B@A      (@I      (@a$G4??Q?idX?B????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      2@9      2@A      (@I      (@a$G4??Q?i???k(???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      &@9      &@A      &@I      &@a-5XE?O?i??p[S0???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?j?ӕ?I?in?? ?6???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a?	???F?i0Zj??<???Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_14/MatMul(1       @9       @A       @I       @a?	???F?i???_JB???Unknown
?HostMatMul",gradient_tape/sequential_4/dense_14/MatMul_1(1       @9       @A       @I       @a?	???F?i?'s
H???Unknown
tHost_FusedMatMul"sequential_4/dense_14/Relu(1       @9       @A       @I       @a?	???F?iv????M???Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a?	???F?i8?{n?S???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @9      @A      @I      @aU?gO?D?i"?(?X???Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @aU?gO?D?i????]???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aU?gO?D?i??7??b???Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @aU?gO?D?i?\?T?g???Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @aU?gO?D?i?6_?l???Unknown
~ HostMatMul"*gradient_tape/sequential_4/dense_13/MatMul(1      @9      @A      @I      @aU?gO?D?i??Ƿq???Unknown
w!Host_FusedMatMul"sequential_4/dense_15/BiasAdd(1      @9      @A      @I      @aU?gO?D?i?ꆁ?v???Unknown
?"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a$G4??A?i?7*E{???Unknown
e#Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a$G4??A?i?_???Unknown?
?$HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a$G4??A?i??p̮????Unknown
v%HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a???m?<?i?#?F????Unknown
?&HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @a???m?<?iFR?gފ???Unknown
?'HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @a???m?<?i?5v????Unknown
?(HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @a???m?<?i??;????Unknown
?)Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @a???m?<?i???Х????Unknown
?*HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a???m?<?i*S??=????Unknown
?+HostBiasAddGrad"7gradient_tape/sequential_4/dense_13/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???m?<?icTl՜???Unknown
?,HostBiasAddGrad"7gradient_tape/sequential_4/dense_14/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a???m?<?i??:m????Unknown
?-HostMatMul",gradient_tape/sequential_4/dense_15/MatMul_1(1      @9      @A      @I      @a???m?<?iՓ?????Unknown
v.HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?	???6?i6?{??????Unknown
[/HostPow"
Adam/Pow_1(1      @9      @A      @I      @a?	???6?i??=?ĩ???Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a?	???6?i?- ??????Unknown
?1HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a?	???6?iYa?f?????Unknown
v2HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a?	???6?i???>d????Unknown
b3HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?	???6?i?FD????Unknown
?4HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a?	???6?i|??#????Unknown
e5HostAddN"Adam/gradients/AddN(1      @9      @A      @I      @a$G4??1?i???K????Unknown
t6HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a$G4??1?i?H??s????Unknown
V7HostCast"Cast(1      @9      @A      @I      @a$G4??1?i?}??????Unknown
\8HostGreater"Greater(1      @9      @A      @I      @a$G4??1?i??Ou?????Unknown
?9HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A      @I      @a$G4??1?i)<!W?????Unknown
?:HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a$G4??1?i???8????Unknown
V;HostSum"Sum_2(1      @9      @A      @I      @a$G4??1?i;??;????Unknown
j<HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a$G4??1?i?/??b????Unknown
z=HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a$G4??1?iM?gފ????Unknown
?>HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a$G4??1?i?|9??????Unknown
??HostReadVariableOp"+sequential_4/dense_13/MatMul/ReadVariableOp(1      @9      @A      @I      @a$G4??1?i_#??????Unknown
]@HostCast"Adam/Cast_1(1       @9       @A       @I       @a?	???&?i=??J????Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a?	???&?i?V?y?????Unknown
XBHostCast"Cast_3(1       @9       @A       @I       @a?	???&?irp?e*????Unknown
XCHostEqual"Equal(1       @9       @A       @I       @a?	???&?i#??Q?????Unknown
VDHostMean"Mean(1       @9       @A       @I       @a?	???&?iԣp=
????Unknown
rEHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a?	???&?i??Q)z????Unknown
vFHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a?	???&?i6?2?????Unknown
?GHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a?	???&?i??Z????Unknown
vHHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a?	???&?i?
???????Unknown
}IHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a?	???&?iI$??9????Unknown
`JHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a?	???&?i?=?ĩ????Unknown
?KHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a?	???&?i?W??????Unknown
~LHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a?	???&?i\qy??????Unknown
?MHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a?	???&?i?Z??????Unknown
?NHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a?	???&?i??;ti????Unknown
~OHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a?	???&?io?`?????Unknown
?PHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?	???&?i ??KI????Unknown
?QHostReluGrad",gradient_tape/sequential_4/dense_13/ReluGrad(1       @9       @A       @I       @a?	???&?i???7?????Unknown
?RHostReluGrad",gradient_tape/sequential_4/dense_14/ReluGrad(1       @9       @A       @I       @a?	???&?i??#)????Unknown
?SHostReadVariableOp",sequential_4/dense_15/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a?	???&?i3%??????Unknown
?THostReadVariableOp"+sequential_4/dense_15/MatMul/ReadVariableOp(1       @9       @A       @I       @a?	???&?i?>??????Unknown
rUHostSigmoid"sequential_4/dense_15/Sigmoid(1       @9       @A       @I       @a?	???&?i?Xc?x????Unknown
~VHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      ??9      ??A      ??I      ??a?	????im?S?0????Unknown
tWHostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a?	????iErD??????Unknown
vXHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a?	????i?4ɠ????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a?	????i??%?X????Unknown
vZHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a?	????i??????Unknown
X[HostCast"Cast_4(1      ??9      ??A      ??I      ??a?	????i????????Unknown
X\HostCast"Cast_5(1      ??9      ??A      ??I      ??a?	????i}2???????Unknown
a]HostIdentity"Identity(1      ??9      ??A      ??I      ??a?	????iU???8????Unknown?
T^HostMul"Mul(1      ??9      ??A      ??I      ??a?	????i-L،?????Unknown
?_HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?	????i?Ȃ?????Unknown
u`HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a?	????i?e?x`????Unknown
waHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?	????i???n????Unknown
ybHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?	????i??d?????Unknown
xcHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a?	????ie?Z?????Unknown
?dHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a?	????i=?{P@????Unknown
?eHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a?	????i&lF?????Unknown
?fHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a?	????i??\<?????Unknown
?gHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a?	????i??M2h????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a?	????i??=( ????Unknown
?iHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a?	????iuY.?????Unknown
?jHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a?	????iM??????Unknown
?kHostReadVariableOp",sequential_4/dense_14/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a?	????i%s
H????Unknown
?lHostReadVariableOp"+sequential_4/dense_14/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a?	????i?????????Unknown
JmHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
WnHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i?????????Unknown
[oHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i?????????Unknown
YpHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i?????????Unknown*?j
uHostFlushSummaryWriter"FlushSummaryWriter(1     ??@9     ??@A     ??@I     ??@aa?"??ia?"???Unknown?
~HostMatMul"*gradient_tape/sequential_4/dense_15/MatMul(1     ?D@9     ?D@A     ?D@I     ?D@aNy?^wt??i???DR????Unknown
?HostBiasAddGrad"7gradient_tape/sequential_4/dense_15/BiasAdd/BiasAddGrad(1      C@9      C@A      C@I      C@a??w?넘?i??Ӡy????Unknown
iHostWriteSummary"WriteSummary(1      @@9      @@A      @@I      @@a??ӥ??i??;??_???Unknown?
tHost_FusedMatMul"sequential_4/dense_13/Relu(1      =@9      =@A      =@I      =@ac???G???i?H?~Z????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      9@9      9@A      9@I      9@a?*-x?!??i????fv???Unknown
vHostMul"%binary_crossentropy/logistic_loss/mul(1      7@9      7@A      7@I      7@aہ?v`???i??fl ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1      2@9      2@A      2@I      2@a?p?F?:??i?5??
J???Unknown
?	HostReadVariableOp",sequential_4/dense_13/BiasAdd/ReadVariableOp(1      0@9      0@A      0@I      0@a??ӥ??i+P???????Unknown
?
HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      2@9      2@A      .@I      .@aL36?v[??i?(??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1      ,@9      ,@A      ,@I      ,@az?eS??i?C4T2???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      *@9      *@A      *@I      *@a?_??ƀ?i??$ou???Unknown
oHostReadVariableOp"Adam/ReadVariableOp(1      *@9      *@A      *@I      *@a?_??ƀ?ik??????Unknown
~HostSelect"*binary_crossentropy/logistic_loss/Select_1(1      *@9      *@A      *@I      *@a?_??ƀ?i??R?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      (@9      (@A      (@I      (@a?뉳??~?irԹ??9???Unknown
YHostPow"Adam/Pow(1      (@9      (@A      (@I      (@a?뉳??~?iI? ??w???Unknown
dHostDataset"Iterator::Model(1     ?B@9     ?B@A      (@I      (@a?뉳??~?i ??wy????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      2@9      2@A      (@I      (@a?뉳??~?i???j????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      &@9      &@A      &@I      &@a	?9d|?i'?b?2,???Unknown
|HostSelect"(binary_crossentropy/logistic_loss/Select(1      "@9      "@A      "@I      "@a?p?F?:w?i	1??Z???Unknown
^HostGatherV2"GatherV2(1       @9       @A       @I       @a??ӥt?iC>???????Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_14/MatMul(1       @9       @A       @I       @a??ӥt?i}K$e?????Unknown
?HostMatMul",gradient_tape/sequential_4/dense_14/MatMul_1(1       @9       @A       @I       @a??ӥt?i?X??????Unknown
tHost_FusedMatMul"sequential_4/dense_14/Relu(1       @9       @A       @I       @a??ӥt?i?eX??????Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a??ӥt?i+s?[")???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      @9      @A      @I      @az?eSr?i?>??DM???Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @az?eSr?iQ
@?fq???Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @az?eSr?i?????????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      G@9      G@A      @I      @az?eSr?iw??&?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @az?eSr?i
m4Y?????Unknown
~HostMatMul"*gradient_tape/sequential_4/dense_13/MatMul(1      @9      @A      @I      @az?eSr?i?8ۋ????Unknown
w Host_FusedMatMul"sequential_4/dense_15/BiasAdd(1      @9      @A      @I      @az?eSr?i0??&???Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?뉳??n?i?5|
E???Unknown
e"Host
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?뉳??n?i?9d???Unknown?
?#HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1      @9      @A      @I      @a?뉳??n?i?????????Unknown
v$HostSum"%binary_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @aeDH?H?i?i8?\@˜???Unknown
?%HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1      @9      @A      @I      @aeDH?H?i?i|2??????Unknown
?&HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1      @9      @A      @I      @aeDH?H?i?i?z??i????Unknown
?'HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1      @9      @A      @I      @aeDH?H?i?iÝ9????Unknown
?(Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1      @9      @A      @I      @aeDH?H?i?iH^c???Unknown
?)HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @aeDH?H?i?i?S?????Unknown
?*HostBiasAddGrad"7gradient_tape/sequential_4/dense_13/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aeDH?H?i?iЛ???7???Unknown
?+HostBiasAddGrad"7gradient_tape/sequential_4/dense_14/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aeDH?H?i?i??=vQ???Unknown
?,HostMatMul",gradient_tape/sequential_4/dense_15/MatMul_1(1      @9      @A      @I      @aeDH?H?i?iX,_?Ek???Unknown
v-HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a??ӥd?i?2,Z????Unknown
[.HostPow"
Adam/Pow_1(1      @9      @A      @I      @a??ӥd?i?9?-?????Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a??ӥd?i/@?7????Unknown
?0HostCast"Tbinary_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1      @9      @A      @I      @a??ӥd?i?F??ܽ???Unknown
v1HostNeg"%binary_crossentropy/logistic_loss/Neg(1      @9      @A      @I      @a??ӥd?iiM`??????Unknown
b2HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a??ӥd?iT-}(????Unknown
?3HostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1      @9      @A      @I      @a??ӥd?i?Z?P?????Unknown
e4HostAddN"Adam/gradients/AddN(1      @9      @A      @I      @a?뉳??^?i?ԯJ???Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?뉳??^?i???????Unknown
V6HostCast"Cast(1      @9      @A      @I      @a?뉳??^?i???mC*???Unknown
\7HostGreater"Greater(1      @9      @A      @I      @a?뉳??^?i{na̿9???Unknown
?8HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      5@9      5@A      @I      @a?뉳??^?iq3;+<I???Unknown
?9HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      ??A      @I      ??a?뉳??^?ig???X???Unknown
V:HostSum"Sum_2(1      @9      @A      @I      @a?뉳??^?i]???4h???Unknown
j;HostMean"binary_crossentropy/Mean(1      @9      @A      @I      @a?뉳??^?iS??G?w???Unknown
z<HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1      @9      @A      @I      @a?뉳??^?iIG??-????Unknown
?=HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1      @9      @A      @I      @a?뉳??^?i?|?????Unknown
?>HostReadVariableOp"+sequential_4/dense_13/MatMul/ReadVariableOp(1      @9      @A      @I      @a?뉳??^?i5?Ud&????Unknown
]?HostCast"Adam/Cast_1(1       @9       @A       @I       @a??ӥT?i?T<Ny????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_4(1       @9       @A       @I       @a??ӥT?i??"8̺???Unknown
XAHostCast"Cast_3(1       @9       @A       @I       @a??ӥT?i"[	"????Unknown
XBHostEqual"Equal(1       @9       @A       @I       @a??ӥT?iq??r????Unknown
VCHostMean"Mean(1       @9       @A       @I       @a??ӥT?i?a???????Unknown
rDHostAdd"!binary_crossentropy/logistic_loss(1       @9       @A       @I       @a??ӥT?i???????Unknown
vEHostExp"%binary_crossentropy/logistic_loss/Exp(1       @9       @A       @I       @a??ӥT?i^h??j????Unknown
?FHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1       @9       @A       @I       @a??ӥT?i?뉳?????Unknown
vGHostSub"%binary_crossentropy/logistic_loss/sub(1       @9       @A       @I       @a??ӥT?i?np????Unknown
}HHostDivNoNan"'binary_crossentropy/weighted_loss/value(1       @9       @A       @I       @a??ӥT?iK?V?c???Unknown
`IHostDivNoNan"
div_no_nan(1       @9       @A       @I       @a??ӥT?i?u=q????Unknown
?JHostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1       @9       @A       @I       @a??ӥT?i??#[	"???Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1       @9       @A       @I       @a??ӥT?i8|
E\,???Unknown
?LHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1       @9       @A       @I       @a??ӥT?i???.?6???Unknown
?MHost	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1       @9       @A       @I       @a??ӥT?iւ?A???Unknown
~NHostRealDiv")gradient_tape/binary_crossentropy/truediv(1       @9       @A       @I       @a??ӥT?i%?UK???Unknown
?OHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a??ӥT?it????U???Unknown
?PHostReluGrad",gradient_tape/sequential_4/dense_13/ReluGrad(1       @9       @A       @I       @a??ӥT?i????_???Unknown
?QHostReluGrad",gradient_tape/sequential_4/dense_14/ReluGrad(1       @9       @A       @I       @a??ӥT?i?q?Mj???Unknown
?RHostReadVariableOp",sequential_4/dense_15/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a??ӥT?iaX??t???Unknown
?SHostReadVariableOp"+sequential_4/dense_15/MatMul/ReadVariableOp(1       @9       @A       @I       @a??ӥT?i??>??~???Unknown
rTHostSigmoid"sequential_4/dense_15/Sigmoid(1       @9       @A       @I       @a??ӥT?i?%~F????Unknown
~UHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      ??9      ??A      ??I      ??a??ӥD?i?[?o????Unknown
tVHostReadVariableOp"Adam/Cast/ReadVariableOp(1      ??9      ??A      ??I      ??a??ӥD?iM?h?????Unknown
vWHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a??ӥD?i???????Unknown
vXHostAssignAddVariableOp"AssignAddVariableOp_1(1      ??9      ??A      ??I      ??a??ӥD?i? ?Q?????Unknown
vYHostAssignAddVariableOp"AssignAddVariableOp_3(1      ??9      ??A      ??I      ??a??ӥD?iBb??????Unknown
XZHostCast"Cast_4(1      ??9      ??A      ??I      ??a??ӥD?i???;?????Unknown
X[HostCast"Cast_5(1      ??9      ??A      ??I      ??a??ӥD?i??˰h????Unknown
a\HostIdentity"Identity(1      ??9      ??A      ??I      ??a??ӥD?i7'?%?????Unknown?
T]HostMul"Mul(1      ??9      ??A      ??I      ??a??ӥD?i?h???????Unknown
?^HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a??ӥD?i????????Unknown
u_HostReadVariableOp"div_no_nan/ReadVariableOp(1      ??9      ??A      ??I      ??a??ӥD?i,옄????Unknown
w`HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??ӥD?i?-??7????Unknown
yaHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a??ӥD?izona????Unknown
xbHostCast"&gradient_tape/binary_crossentropy/Cast(1      ??9      ??A      ??I      ??a??ӥD?i!?r??????Unknown
?cHostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1      ??9      ??A      ??I      ??a??ӥD?i??eX?????Unknown
?dHost
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1      ??9      ??A      ??I      ??a??ӥD?io4Y??????Unknown
?eHostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1      ??9      ??A      ??I      ??a??ӥD?ivLB????Unknown
?fHostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1      ??9      ??A      ??I      ??a??ӥD?i????0????Unknown
?gHostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1      ??9      ??A      ??I      ??a??ӥD?id?2,Z????Unknown
?hHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1      ??9      ??A      ??I      ??a??ӥD?i;&??????Unknown
?iHostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1      ??9      ??A      ??I      ??a??ӥD?i?|?????Unknown
?jHostReadVariableOp",sequential_4/dense_14/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a??ӥD?iY???????Unknown
?kHostReadVariableOp"+sequential_4/dense_14/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a??ӥD?i      ???Unknown
JlHostReadVariableOp"div_no_nan_1/ReadVariableOp(i      ???Unknown
WmHostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(i      ???Unknown
[nHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(i      ???Unknown
YoHostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(i      ???Unknown2CPU