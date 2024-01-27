import { z } from "zod";
import {
  createTRPCRouter,
  protectedProcedure,
  publicProcedure,
} from "~/server/api/trpc";
import { hash, verify } from "argon2";
import { TRPCError } from "@trpc/server";

export const userRouter = createTRPCRouter({
  getUser: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const user = await ctx.db.users.findFirst({
        where: { id: input.id },
      });
      return user;
    }),

  createUser: publicProcedure
    .input(
      z.object({
        email: z.string().email(),
        password: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      //check first if email exists, throw error
      const exist = await ctx.db.user.findFirst({
        where: {
          email: input.email,
        },
      });
      if (exist) {
        throw new TRPCError({
          code: "BAD_REQUEST",
          message: "Email already exists",
        });
      }

      const hashedPassword = await hash(input.password); // Hash the password
      const newUser = await ctx.db.user.create({
        data: {
          email: input.email,
          password: hashedPassword,
        },
      });
      //sign in User for next-auth session
      return newUser;
    }),
  loginUser: publicProcedure
    .input(
      z.object({
        email: z.string().email(),
        password: z.string(),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const logUser = await ctx.db.user.findFirst({
        where: {
          email: input.email,
        },
      });
      const pswd = logUser?.password as string;
      const isValidPassword = await verify(pswd, input.password);
      if (!isValidPassword) {
        throw new TRPCError({
          code: "UNAUTHORIZED",
          message: "Invalid Email or Password",
        });
      }
      return logUser;
    }),
});
